import os
import re
import cv2
import json
import torch
import pandas as pd
import numpy as np
import config as c
import matplotlib.pyplot as plt
import staintools
from PIL import Image
from collections import Counter
from matplotlib.ticker import FuncFormatter
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
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

def setup_env_variables():
    # Determine the available GPU memory
    num_gpus = torch.cuda.device_count()
    available_memory_gb = []

    for gpu_id in range(num_gpus):
        gpu = torch.cuda.get_device_properties(gpu_id)
        available_memory_gb.append(gpu.total_memory / (1024 ** 3))

    # Calculate the maximum memory usage per GPU
    max_memory_gb = max(available_memory_gb)  # Use the GPU with the most memory
    max_split_size_mb = int(max_memory_gb * 1024)  # Convert to MB

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{max_split_size_mb}"


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

def load_paths_labels_annotations(wsi_folders, labels_dict):
    wsi_paths = []
    labels = []
    annotations = []
    slides_skipped = []

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
                    slides_skipped.append(pathology_num)
                    continue 

                wsi_paths.append(wsi_path)
                labels.append(label)

                # Assuming the annotation files are named as "<wsi_file>.geojson"
                annotation_path = os.path.join(c.ANNOTATIONS_PATH, f"{os.path.splitext(file)[0]}.geojson")
                
                annotation = None
                if os.path.exists(annotation_path):
                    with open(annotation_path, "r") as f:
                        annotation = json.load(f)
                annotations.append(annotation)

    if slides_skipped:
        print(f'Label was not fround for {len(slides_skipped)} slides. Slides that were skipped: {slides_skipped}')

    return wsi_paths, labels, annotations

def load_predictions_features(predictions_path, features_path):
    '''Returns a dataframe by horizontally concatenating the predictions and features.'''
    preds = pd.read_excel(predictions_path)
    features = pd.DataFrame(pd.read_pickle(features_path))

    # Concatenate dataframes horizontally
    return pd.concat([preds, features.drop(columns=["wsi_id", "patch_index"])], axis=1)

def load_avg_predictions_features(predictions_path, features_path):
    '''Returns a dataframe by merging the predictions with the features based on the WSI id and patch index, averaging duplicates.'''
    preds = pd.read_excel(predictions_path)
    features = pd.DataFrame(pd.read_pickle(features_path))

    # Average the probabilities for preds with the same wsi_id and patch_index (duplicates in the training set result of data augmentation)
    avg_preds = preds.groupby(['wsi_id', 'patch_index']).mean().reset_index()

    # Average the features for features with the same wsi_id and patch_index
    avg_features = features.groupby(['wsi_id', 'patch_index']).agg(lambda x: list(np.mean(np.vstack(x), axis=0))).reset_index()

    # Merge the averaged predictions and features
    patches_df = avg_preds.merge(avg_features, on=["wsi_id", "patch_index"]) # merge tables horizontally

    return patches_df

def get_mrn_for_pathology_num(pathology_num, df, columns):
    '''Returns the MRN for a given pathology number, by looking for the pathology number within the specified columns of df.'''

    for col in columns:
        mrn = df.loc[df[col] == pathology_num]['MRN'].values
        if len(mrn) > 0:
            return int(mrn[0])
        
    return None
        

# --------------------------------------------------------------------------------
# WSI properties methods
# --------------------------------------------------------------------------------

def get_pathology_num_from_labels(slide_id, labels_dict = None, match_labels = False, separator = ' '):
    slide_id = os.path.basename(slide_id).split(separator)[0].strip().split()[0] # e.g. "S60-1234 A2.svs" => "S60-1234"

    if labels_dict and slide_id in labels_dict.keys():
        return slide_id
    
    # If the name matches the format "60S1234"
    if re.match(r"^\d+[A-Za-z]+-\d+$", slide_id) or re.match(r"^\d+[A-Za-z]+\d+$", slide_id):
        prefix_number = re.findall(r"^\d+", slide_id)[0]
        letter = re.findall(r"[A-Za-z]+", slide_id)[0]
        postfix_number = re.findall(r"\d+$", slide_id)[0]
        return f"{letter}{prefix_number.zfill(2)}-{postfix_number.zfill(4)}"
    
    if labels_dict and match_labels and re.match(r"^[A-Za-z]+\d+-\d+$", slide_id): # e.g. img_name is S60-123, also check for S60-0123 in Labels.xlsx
        first_part = slide_id[:slide_id.index('-')]
        second_part = slide_id[slide_id.index('-') + 1:]
        return f"{first_part}-{second_part.zfill(4)}" # e.g. S60-123 => S60-0123

    return slide_id

def get_file_name_from_pathology_num(pathology_num, dir_path = c.LGG_WSI_PATH, file_type = 'svs'):
    slide_files_names =  [file for file in os.listdir(dir_path) if file.endswith(file_type)]
    file_name = next((file for file in slide_files_names if file.startswith(pathology_num)), None)
    if file_name is not None:
        return file_name  
    
    ind =  pathology_num.index('-')
    adj_pathology_num = pathology_num[:ind+1]+pathology_num[ind+1:].lstrip("0")
    file_name = next((file for file in slide_files_names if file.startswith(adj_pathology_num)), None) # e.g. S60-0023 => S60-23
    if file_name is not None:
        return file_name  
    else:
        print(f'Could not find the correspnding slide image for: {pathology_num} in {c.LGG_WSI_PATH}. Excluding this file from the analysis...')
        return None


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
# Get WSI info methods
# --------------------------------------------------------------------------------

def check_duplicate_slides(folders_paths, case_match=True):
    """
    Checks if there are duplicate slides in any of the folders before patch extraction to avoid having two slides in different sets.
    Used when we want the analysis to only include one slide per case

    Args:
        folders_paths (list): List of paths to folders containing the WSIs.
        case_match (bool): Whether to check for exact matches in file names, or check only matches in case ID. 
            If True, only checks if the first part of the slide name matches (case id only) (e.g. S00-1234 A1.svs == S00-1234 B1.svs).
            If False, checks the entire slide name (case id + block id...) (e.g. S00-1234 A1.svs != S00-1234 B1.svs).

    Returns:
        dict: A dictionary containing the duplicate slides.
    """
   
    slide_dict = {}

    for folder in folders_paths:
        for slide in [x for x in os.listdir(folder) if x.endswith('.svs')]:

            if case_match:
                slide_name = get_pathology_num_from_labels(slide) # Only check matches in case ID
            else:
                slide_name = slide

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
        return True
    
    return False

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

def is_valid_patch_list(dir_path, starts_with):
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

    elements = [str(e) if not isinstance(e, (float, np.float64)) else 'N/A' for e in element_count.keys()]
    # elements = [e if len(e)<20 else e[:20]+'..' for e in elements]

    frequencies = list(element_count.values())

    plt.bar(elements, frequencies)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title.capitalize())
    plt.xticks(rotation=90)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

    if log_scale:
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(plot_path)

if __name__ == "__main__":

    setup_directories()
    print("Created patch directories.") 