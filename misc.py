import os
import re
import cv2
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import staintools

from PIL import Image
from collections import Counter
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight

import config as c
from normalize_staining import normalizeStaining

ref_patch = np.array(Image.open(c.STAIN_NORMALIZATION_REF))

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
                pathology_num = get_pathology_num_from_labels(file, labels_dict, match_labels = True, separator=' ')
                if pathology_num in labels_dict.keys():
                    label = labels_dict[pathology_num]
                else:      
                    print(f'Label was not found for {pathology_num}. Skipping this slide...')
                    continue 

                wsi_paths.append(wsi_path)
                labels.append(label)

                # Assuming the annotation files are named as "<wsi_file>.geojson"
                annotation_path = os.path.join(folder, f"{os.path.splitext(file)[0]}.geojson")
                
                annotation = None
                if os.path.exists(annotation_path):
                    with open(annotation_path, "r") as f:
                        annotation = json.load(f)
                annotations.append(annotation)

    return wsi_paths, labels, annotations

def load_predictions_features(predictions_path, features_path):
    '''Returns a dataframe by merging the predictions with the features based on the WSI id and patch index.'''
    preds = pd.read_excel(predictions_path)
    features = pd.DataFrame(pd.read_pickle(features_path))

    patches_df = preds.merge(features, on=["wsi_id", "patch_index"]) # merge tables horizantally
    return patches_df

def get_mrn_for_pathology_num(pathology_num, df, columns):
    '''Returns the MRN for a given pathology number, by looking for the pathology number within the specified columns of df.'''

    for col in columns:
        mrn = df.loc[df[col] == pathology_num]['MRN'].values
        if len(mrn) > 0:
            return int(mrn[0])
        
    return None
        

# --------------------------------------------------------------------------------
# WSI properties methdods
# --------------------------------------------------------------------------------

def get_pathology_num_from_labels(img_name, labels_dict = None, match_labels = False, separator = ' '):
    img_name = os.path.basename(img_name).split(separator)[0].strip() # e.g. "S60-1234 A2.svs" => "S60-1234"

    if labels_dict and img_name in labels_dict.keys():
        return img_name
    
    # If the name matches the format "60S1234"
    if re.match(r"^\d+[A-Za-z]+-\d+$", img_name) or re.match(r"^\d+[A-Za-z]+\d+$", img_name):
        prefix_number = re.findall(r"^\d+", img_name)[0]
        letter = re.findall(r"[A-Za-z]+", img_name)[0]
        postfix_number = re.findall(r"\d+$", img_name)[0]
        return f"{letter}{prefix_number.zfill(2)}-{postfix_number.zfill(4)}"
    
    if labels_dict and match_labels and re.match(r"^[A-Za-z]+\d+-\d+$", img_name): # e.g. img_name is S60-123, also check for S60-0123 in Labels.xlsx
        first_part = img_name[:img_name.index('-')]
        second_part = img_name[img_name.index('-') + 1:]
        return f"{first_part}-{second_part.zfill(4)}" # e.g. S60-123 => S30-0123

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

def check_duplicate_slides(folders_paths):
    """Checks if there are duplicate slides in any of the folders before patch extraction (same case different blocks)."""
   
    slide_dict = {}

    for folder in folders_paths:
        for slide in [x for x in os.listdir(folder) if x.endswith('.svs')]:
            slide_name = get_pathology_num_from_labels(slide)
            path = os.path.join(folder, slide)

            if slide_name not in slide_dict:
                slide_dict[slide_name] = [path]
            else:
                slide_dict[slide_name].append(path)

    duplicates = {slide: paths for slide, paths in slide_dict.items() if len(paths) > 1}

    if duplicates:
        print("There are", len(duplicates), "duplicate slides as follows:")
        for slide, paths in duplicates.items():
            print("Slide:", slide)
            print("Found in paths:", paths)

    return duplicates

def get_wsi_ids_labels(folders_paths, labels, file_type = 'svs', seperator = ' '):
    wsi_ids = {}
    for folder in folders_paths:
        for slide in [x for x in os.listdir(folder) if x.endswith(file_type)]:
            pathology_num = get_pathology_num_from_labels(slide, labels, match_labels=True, separator=seperator)
            
            label = None
            if labels and pathology_num in labels.keys():
                label = c.CLASSES.index(labels[pathology_num])

            if label is not None:
                if label in wsi_ids and pathology_num not in wsi_ids[label]:
                    wsi_ids[label].append(pathology_num)
                elif label not in wsi_ids:
                    wsi_ids[label] = [pathology_num]

    for label in wsi_ids:
        print(f'{c.CLASSES[label]}: {len(wsi_ids[label])} samples')

    return wsi_ids 

def get_class_weights(patches_paths, labels_dict, file_type='svs', seperator=' '):
    ids_labels_dict = get_wsi_ids_labels(patches_paths, labels_dict, file_type=file_type, seperator=seperator)

    class_labels = [label for label, slide_ids in ids_labels_dict.items() for _ in range(len(slide_ids))]

    class_weights = compute_class_weight('balanced', classes=np.unique(list(ids_labels_dict.keys())), y=class_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    return class_weights

# --------------------------------------------------------------------------------
# Patch preprocessing methods
# --------------------------------------------------------------------------------

def apply_stain_normalization(patch):
    patch = np.array(patch)  # Convert the PIL.Image object to a NumPy array
    normalized_patch, _, _ = normalizeStaining(patch)  # Perform H&E stain normalization
    return Image.fromarray(normalized_patch)

def apply_stain_normalization_vahadane(patch):
    try:
        # Temporarily set 'np.bool' to 'np.bool_' to ensure compatibility with the CURRENT (latest) 
        # version of SPAMS, which relies on the deprecated 'np.bool' attribute.
        np.bool = np.bool_
        METHOD = 'vahadane'    
        target = staintools.read_image(c.STAIN_NORMALIZATION_REF)

        normalizer = staintools.StainNormalizer(method=METHOD)
        normalizer.fit(target)
        return normalizer.transform(np.array(patch))
    finally:
        np.bool = None 

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

def is_valid_patch(patch, min_pixel_mean=50, max_pixel_mean=230, max_pixel_min=95, min_std_dev=10, min_saturation=25, max_value=240, otsu_ratio_threshold=0.55):
    # NOTE: patch must be normalized (0 - 255) before calling this function, otherwise it won't work

    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    s = hsv_patch[:, :, 1]
    v = hsv_patch[:, :, 2]
    
    # Otsu's thresholding
    _, otsu_thresh = cv2.threshold(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_ratio = (otsu_thresh == 0).sum() / (otsu_thresh.shape[0] * otsu_thresh.shape[1]) # Ratio of dark pixels
    
    return (min_pixel_mean < patch.mean() < max_pixel_mean and 
            patch.min() < max_pixel_min and 
            patch.std() > min_std_dev and
            s.mean() > min_saturation and
            v.mean() < max_value and  # Filter out very dark patches
            otsu_ratio < otsu_ratio_threshold)  # Ensure that the ratio of dark pixels is below a certain threshold

def is_valid_patches(dir_path, starts_with):
    files = [f for f in os.listdir(dir_path) if f.startswith(starts_with)]
    # files=['pen11.png']
    for file_name in files:
        file_name = os.path.join(dir_path, file_name)
        im = Image.open(file_name)
        im = np.array(im)

        if is_valid_patch(im):
            print(file_name, 'is a valid patch.')
            pass
        else:
            print(file_name, 'is not a valid patch')

# Checks if a patch does not have enough tissue regions so it can be discarded.
def is_white_background(patch, white_ratio_threshold=0.2, subpatch_size=32, mean_rgb_threshold=230, std_rgb_threshold=20):

    patch_normalized = ((patch - patch.min()) / (patch.max() - patch.min())) * 255

    # Slide a window of size "subpatch_size" over the patch and count how many sub-patches are white
    white_subpatches = 0
    total_subpatches = 0

    for i in range(0, patch.shape[0], subpatch_size):
        for j in range(0, patch.shape[1], subpatch_size):
            subpatch = patch_normalized[i:i+subpatch_size, j:j+subpatch_size]
            mean_rgb = np.mean(subpatch, axis=(0, 1))
            std_rgb = np.std(subpatch, axis=(0, 1))
            
            if np.all(mean_rgb > mean_rgb_threshold) and np.all(std_rgb < std_rgb_threshold):
                white_subpatches += 1
            
            total_subpatches += 1

    white_ratio = white_subpatches / total_subpatches
    return white_ratio > white_ratio_threshold  # Reject if more than 20% of the sub-patches are white

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

def save_stats_plots(data, plot_path, title='', xlabel='', ylabel='', log_scale = False):

    element_count = Counter(data)

    elements = list(element_count.keys())
    frequencies = list(element_count.values())

    plt.bar(elements, frequencies)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=90)
    if log_scale:
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(plot_path)

if __name__ == "__main__":
    pass
