import re
import os
import normalizeStaining
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, remove_small_holes
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern

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
    
def get_base_magnification(wsi_img):
    try:
        base_magnification = float(wsi_img.properties['openslide.objective-power'])
    except KeyError:
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

# --------------------------------------------------------------------------------
# Patch Validatoin Methods
# --------------------------------------------------------------------------------

def is_valid_patch(patch, min_pixel_mean=50, max_pixel_mean=230, max_pixel_min=95):
    # patch must be normalized (0 - 255)
    return min_pixel_mean < patch.mean() < max_pixel_mean and patch.min() < max_pixel_min

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
def has_enough_tissue(patch, tissue_percent=80.0, near_zero_var_threshold=0.1, white_threshold=220, uniform_threshold=0.5):
    
    # Convert to grayscale
    grayscale_patch = rgb2gray(patch)

    # Apply color thresholding to filter out the white background
    # This is done on the grayscale image to simplify the process
    mask = grayscale_patch < white_threshold / 255.0 # normalize to [0, 1] range for skimage images
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
    lbp = local_binary_pattern(grayscale_patch, 8, 1)  # Number of points in the circular LBP and radius are set to default values
    lbp_var = np.var(lbp)

    return enough_tissue and lbp_var > uniform_threshold