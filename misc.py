import os
import re
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, remove_small_holes
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern

import config as c
from normalize_staining import normalizeStaining

# --------------------------------------------------------------------------------
# Set up
# --------------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def setup_directories():
    directories = [c.FILES_PATH, c.PATCHES_PATH, c.TRAIN_PATCHES, c.VALID_PATCHES, c.TEST_PATCHES]
    for dir in directories:
        os.makedirs(dir, exist_ok=True)

# --------------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------------

def load_paths(dir_path):
    wsi_paths = []

    for file in os.listdir(dir_path):
        if file.endswith('.svs'):
            wsi_path = os.path.join(dir_path, file)
            wsi_paths.append(wsi_path)

    return wsi_paths

def load_paths_labels_annotations(wsi_folders, labels_file):
    wsi_paths = []
    labels = []
    annotations = []

    # Load labels from an Excel file
    df = pd.read_excel(labels_file)
    labels_dict = dict(zip(df['Pathology Number'].str.strip(), df['Label'].str.strip()))

    # Each folder has both svs files of the WSIs, and geojson files of the corresponding annotations
    # NOTE: both svs and geojson files should have the same name in order for the WSIs be mapped correctly to their annotations
    for folder in wsi_folders:
        for file in os.listdir(folder):
            if file.endswith('.svs'):
                wsi_path = os.path.join(folder, file)
                wsi_paths.append(wsi_path)
                pathology_num = get_pathology_number(file)
                if pathology_num in labels_dict.keys():
                    label = labels_dict[get_pathology_number(file)]
                else:
                    raise ValueError('Label was not found.') 
                labels.append(label)

                # Assuming the annotation files are named as "<wsi_file>.geojson"
                annotation_path = os.path.join(folder, f"{os.path.splitext(file)[0]}.geojson")
                
                if os.path.exists(annotation_path):
                    with open(annotation_path, "r") as f:
                        annotation = json.load(f)
                annotations.append(annotation)

    return wsi_paths, labels, annotations

def load_predictions_features(predictions_path, features_path):
    preds = pd.read_excel(predictions_path)
    features = pd.DataFrame(pd.read_pickle(features_path))

    patches_df = preds.merge(features, on=["wsi_id", "patch_index"]) # merge tables horizantally
    return patches_df

# --------------------------------------------------------------------------------
# WSI properties methdods
# --------------------------------------------------------------------------------

def get_pathology_number(img_name):
    img_name = img_name.split()[0].strip()
    # If the name matches the first format (i.e., "91S5432")
    if re.match(r"^\d+[A-Za-z]+-\d+$", img_name) or re.match(r"^\d+[A-Za-z]+\d+$", img_name):
        prefix_number = re.findall(r"^\d+", img_name)[0]
        letter = re.findall(r"[A-Za-z]+", img_name)[0]
        postfix_number = re.findall(r"\d+$", img_name)[0]
        return f"{letter}{prefix_number.zfill(2)}-{postfix_number.zfill(4)}"
    # If the name matches the second format (i.e., "S12-2343")
    elif re.match(r"^[A-Za-z]+\d+-\d+$", img_name):
        return img_name
    else:
        return img_name


def get_base_magnification(wsi_img, id = ''):
    """
    Retrieve the base magnification level of a WSI.

    Args:
        wsi_img (openslide.OpenSlide): The opened WSI image.

    Returns:
        float: The base magnification level of the WSI.
    """
    try:
        base_magnification = float(wsi_img.properties['openslide.objective-power'])
    except KeyError:
        print(f'Could not find objective power/base magnification level for WSI: {id}. An estimate is done.')
        try:
            mpp_x = float(wsi_img.properties['openslide.mpp-x'])
            mpp_y = float(wsi_img.properties['openslide.mpp-y'])
            # The pixel size at 1x magnification is typically around 0.25 micrometers
            pixel_size_at_1x = 0.25
            base_magnification = 1 / (max(mpp_x, mpp_y) * pixel_size_at_1x)
        except KeyError:
            try:
                highest_res_dim = max(wsi_img.level_dimensions[0])
                lowest_res_dim = max(wsi_img.level_dimensions[-1])
                base_magnification = highest_res_dim / lowest_res_dim
            except:
                return None
    return base_magnification


# --------------------------------------------------------------------------------
# Patch preprocessing methods
# --------------------------------------------------------------------------------

def apply_stain_normalization(patch):
    try:
        patch = np.array(patch)  # Convert the PIL.Image object to a NumPy array
        normalized_patch, _, _ = normalizeStaining(patch)  # Perform H&E stain normalization
        return Image.fromarray(normalized_patch)
    except np.linalg.LinAlgError:
        return patch

def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(apply_stain_normalization),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Standard normalization values since we use pretrained models
    ])

def custom_collate_fn(batch):
    images = []
    coords = []
    patch_idxs = []
    wsi_idxs = []
    labels = []

    for image, coord, patch_idx, wsi_idx, label in batch:
        images.append(image)
        coords.append(coord)
        patch_idxs.append(patch_idx)
        wsi_idxs.append(wsi_idx)
        labels.append(label)

    return images, torch.tensor(coords, dtype=torch.int), torch.tensor(patch_idxs, dtype=torch.int), torch.tensor(wsi_idxs, dtype=torch.int), torch.tensor(labels, dtype=torch.long)

# --------------------------------------------------------------------------------
# Patch Validation Methods
# --------------------------------------------------------------------------------

def is_valid_patch(patch, min_pixel_mean=50, max_pixel_mean=230, max_pixel_min=95):
    # NOTE: patch must be normalized (0 - 255), otherwise it won't work
    return not is_white_background(patch) and (min_pixel_mean < patch.mean() < max_pixel_mean and patch.min() < max_pixel_min)


# Checks if a patch does not have enough tissue regions so it can be discarded.
# TODO: Implement other preprocessing steps (e.g. is_contain_artifacts)
def is_white_background(patch):
    patch_array = np.array(patch)
    mean_rgb = np.mean(patch_array, axis=(0, 1))

    threshold = 200
    is_white = np.all(mean_rgb > threshold)

    return is_white


# Checks if a patch does not have enough tissue regions so it can be discarded.
# TODO: Implement other preprocessing steps (e.g. is_contain_artifacts)
def has_enough_tissue(patch, tissue_percent=80.0, near_zero_var_threshold=0.1, white_threshold=220,
                      uniform_threshold=0.5):
    # Convert to grayscale
    grayscale_patch = rgb2gray(patch)

    # Apply color thresholding to filter out the white background
    # This is done on the grayscale image to simplify the process
    mask = grayscale_patch < white_threshold / 255.0  # normalize to [0, 1] range for skimage images
    grayscale_patch = grayscale_patch * mask

    # Apply Otsu thresholding
    threshold = threshold_otsu(grayscale_patch)
    binary_patch = grayscale_patch > threshold

    # Perform binary dilation and fill small holes
    dilated_patch = binary_dilation(binary_patch)
    filled_patch = remove_small_holes(dilated_patch)

    # Compute variance after morphological operations
    var_after_morphology = np.var(filled_patch)

    # Calculate the percentage of tissue pixels
    tissue_pixels = np.sum(filled_patch)
    total_pixels = np.size(filled_patch)
    tissue_percent_actual = (tissue_pixels / total_pixels) * 100

    # Determine if there is enough tissue
    enough_tissue = tissue_percent_actual > tissue_percent and var_after_morphology > near_zero_var_threshold

    # Implement a check for large uniform areas
    # We use local binary pattern for texture analysis
    lbp = local_binary_pattern(grayscale_patch, 8,
                               1)  # Number of points in the circular LBP and radius are set to default values
    lbp_var = np.var(lbp)

    return enough_tissue and lbp_var > uniform_threshold

# --------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------

def save_roc_auc_plot(fpr, tpr, roc_auc, plot_path):
    n_classes = len(c.CLASSES)

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    # Plot all ROC curves
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(plot_path)
