import os
import cv2
import re
import json
import misc
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from normalizeStaining import normalizeStaining
from sklearn.model_selection import train_test_split
from shapely.geometry import shape, Polygon
from rasterio.features import rasterize

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

openslide_path = r"C:\Users\sarah\Documents\openslide-win64-20230414\bin"
os.environ['PATH'] = openslide_path + ";" + os.environ['PATH']
os.add_dll_directory(openslide_path)
import openslide

class PatchDataset(Dataset):
    def __init__(self, wsi_paths, annotations, target_magnification, base_magnification=20.0, num_patches = 100, base_patch_size=(224, 224), transform=None, save_dir=None):
        self.wsi_paths = wsi_paths
        self.annotations = annotations
        self.target_magnification = target_magnification
        self.base_magnification = base_magnification
        self.num_patches = num_patches
        self.base_patch_size = base_patch_size
        self.transform = transform
        self.save_dir = save_dir # Save the extracted patches after stain normalization
        self.saved_patches_wsi = set()  # Create a set to keep track of the WSIs for which the patches have already been saved

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # Remove previous coords files if exists
        coords_file_path = os.path.join(files_path, "patches_coords.xlsx")
        if os.path.exists(coords_file_path):
            os.remove(coords_file_path)

    def __len__(self):
        return len(self.wsi_paths)

    # Return patches from a WSI
    def __getitem__(self, index):
        wsi_path = self.wsi_paths[index]
        annotation = self.annotations[index]

        # Get the pathology number from the WSI path
        img_name = os.path.basename(wsi_path)
        pathology_number = get_pathology_number(img_name)

        # Check if patches for the current WSI already exist in any of the sets
        # NOTE: Changing folder names in which the patches will be stored requires changing the list
        for patches_path in ['train_patches', 'valid_patches', 'test_patches']:
            if os.path.exists(patches_path):
                patch_files = os.listdir(patches_path)
                if any(file.startswith(pathology_number) for file in patch_files):
                    # Patches already exist for this WSI, so skip to the next
                    print('Patches for', pathology_number, 'already exist. Skipped extracting patches.' )
                    return []

        # Load the WSI image
        wsi_img = openslide.OpenSlide(wsi_path)

        # Retrieve the base magnification from the WSI metadata
        base_magnification = misc.get_base_magnification(wsi_img)
        if base_magnification is None:
            print('Base magnification metadata for', pathology_number, 'is missing. Skipped extracting patches.' )
            return []
                
        # Calculate the patch size for the target magnification
        scale_factor = self.base_magnification / self.target_magnification
        patch_size = tuple(int(scale_factor * dim) for dim in self.base_patch_size)

        downsample_factor = base_magnification / self.target_magnification
        best_level = wsi_img.get_best_level_for_downsample(downsample_factor)
        
        level_downsample = wsi_img.level_downsamples[best_level]
        scaled_patch_size = tuple(int(np.ceil(ps / level_downsample)) for ps in patch_size)

        # Generate the ROI mask from the annotation
        roi_mask = annotation_to_roi_mask(annotation, wsi_img.dimensions[::-1])  # NOTE: dimensions are given in (width, height) but numpy arrays are in (height, width)

        # Extract patches as well as their scaled coordiantes from the ROI 
        patches_coords = extract_patches_within_roi(self, wsi_img, roi_mask, best_level, scaled_patch_size, overlap_percent=0, min_overlap_ratio=0.9, num_patches=self.num_patches)

        # Save the normalized patch if a save directory was provided
        if len(patches_coords) > 0 and self.save_dir is not None and wsi_path not in self.saved_patches_wsi:
            self.saved_patches_wsi.add(wsi_path)  # Add the WSI to the set of saved WSIs, only one patch from each WSI is saved

            patches = []
            coords = []
            all_coords = [] # Coordiantes for all the patches of all WSIs

            patch_index = 1
            for patch_coords in patches_coords:
                patch = patch_coords['patch']
                coord = patch_coords['coord']
                patch_id = f"{pathology_number}_{patch_index}" # NOTE: Changing patch file name format requires changing extracting patch name in ClassificationWSI.py

                patches.append(patch)
                coords.append(coord)
                all_coords.append({'patch_id': patch_id, 'X': coord[0], 'Y': coord[1]})
                
                patch_array = patch.numpy().transpose(1, 2, 0)
                # Normalize pixel values to the range of 0-255
                patch_normalized = ((patch_array - patch_array.min()) / (patch_array.max() - patch_array.min())) * 255
                # Convert to uint8
                patch_normalized_uint8 = patch_normalized.astype('uint8')
                patch_pil = Image.fromarray(patch_normalized_uint8)              
                patch_pil.save(os.path.join(self.save_dir, patch_id + ".png"))
                patch_index += 1

            # Save coordinates to an Excel file
            new_coords_df = pd.DataFrame(all_coords)
            coords_file_path = os.path.join("files","patches_coords.xlsx") # NOTE: changing file name requires chaning it in classification_patches.py

            if os.path.exists(coords_file_path):
                existing_coords_df = pd.read_excel(coords_file_path)
                new_coords_df = pd.concat([existing_coords_df, new_coords_df], ignore_index=True)

            new_coords_df.to_excel(coords_file_path, index=False)

            return patches

def annotation_to_roi_mask(annotation, image_dims):
    """Generates a binary mask of the ROI based on a GeoJSON annotation.
    Args:
        annotation (dict): A dictionary containing the GeoJSON annotation.
        image_dims (tuple): The dimensions of the image (width, height).
    Returns:
        np.array: A 2D numpy array representing the binary mask of the ROI.
    """
    shapes = [(shape(feature['geometry']), 1) for feature in annotation['features']]
    # Rasterize the list of shapes onto a binary mask which represent the ROI as a 2D numpy array, 
    # where pixels within the ROI have a value of 1 and pixels outside the ROI have a value of 0.
    mask = rasterize(shapes, out_shape=image_dims)
    return mask

def annotation_to_roi_boxes(annotation):
    """Extracts the bounding boxes of the region of interest (ROI) from a GeoJSON annotation.
    Treat each annotation as a bounding box and slide a window of size patch_size across this bounding box.
    This approach doesn't ensure that patches don't cross the boundaries of the actual ROI polygons if the ROIs are irregularly shaped.
    
    Args:
        annotation (dict): A dictionary containing the GeoJSON annotation.

    Returns:
        list: A list of bounding boxes representing the ROIs.
    """
    roi_boxes = []
    for feature in annotation['features']:
        geometry = feature['geometry']
        if geometry['type'] == 'Polygon':
            for coords in geometry['coordinates']:
                polygon = Polygon(coords[:-1]) # Convert to Shapely polygon
                roi_boxes.append(polygon.bounds) # Get the bounding box of the polygon

    return roi_boxes

def extract_patches_within_roi(self, wsi_img, roi_mask, best_level, scaled_patch_size, overlap_percent=20, min_overlap_ratio=0.7, num_patches=10):
    """
    Extracts patches from the region of interest (ROI) within a whole-slide image (WSI).

    Args:
        roi_mask (np.array): A binary mask of the ROI, where non-zero values indicate the region of interest.
        best_level (int): The level of the WSI from which to extract the patches. This level provides the desired magnification.
        scaled_patch_size (tuple): The size of the patches to extract, scaled to the target magnification level.
        overlap_percent (int, optional): The desired overlap between patches, specified as a percentage. Defaults to 50.
        min_overlap_ratio (float, optional): The minimum required overlap ratio between a patch and the ROI. Defaults to 0.7 (If at least 70% of the patch lies within the ROI, we accept it).
        num_patches (int, optional): The maximum number of patches to extract.

    Returns:
        list: A list of extracted patches.

    """
    patches_coords = [] # Store a dictionary for each patch, where the key is the patch and the value is its coordiantes

    stride = [int(dim * (1 - overlap_percent / 100)) for dim in scaled_patch_size]  # calculate stride based on the desired overlap
    patch_area = np.prod(scaled_patch_size)
    min_within_ROI = patch_area * min_overlap_ratio

    for y in range(0, roi_mask.shape[0], stride[1]):
        for x in range(0, roi_mask.shape[1], stride[0]):
            roi_overlap = np.sum(roi_mask[y:y+scaled_patch_size[1], x:x+scaled_patch_size[0]])
            if roi_overlap >= min_within_ROI: # Only extract patches that are mostly insdie the ROI based on min_overlap_ratio
                coord = (x, y)

                if best_level < len(wsi_img.level_dimensions):
                    # If the desired level is within the available levels of the WSI
                    patch = np.array(wsi_img.read_region(coord, best_level, scaled_patch_size))[:, :, :3]
                else:
                    # If the desired level is beyond the available levels of the WSI
                    full_res_patch = np.array(wsi_img.read_region(coord, 0, scaled_patch_size))[:, :, :3]
                    patch = cv2.resize(full_res_patch, scaled_patch_size, interpolation=cv2.INTER_LINEAR)

                # Ensure that the patch has the same size as base_patch_size 
                if patch.shape[:2] != self.base_patch_size:
                    # Resize the full-resolution patch to the target size using interpolation
                    patch = cv2.resize(patch, self.base_patch_size, interpolation=cv2.INTER_LINEAR)

                h, w, _ = patch.shape
                if h == self.base_patch_size[1] and w == self.base_patch_size[0]:
                    if self.transform:
                        patch = self.transform(patch)
                    patches_coords.append({'patch': patch, 'coord': coord})

                if len(patches_coords) >= num_patches:
                    break
            if len(patches_coords) >= num_patches:
                break
        if len(patches_coords) >= num_patches:
            break

    return patches_coords

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
    
def load_data(hgg_folder, lgg_folder, labels_file):
    wsi_paths = []
    labels = []
    annotations = []

    # Load labels from an Excel file
    df = pd.read_excel(labels_file)
    labels_dict = dict(zip(df['Pathology Number'].str.strip(), df['Label'].str.strip()))

    # Assuming WSIs are divided into two seperate folders (HGG and LGG)
    # Each folder has both svs files of the WSIs, and geojson files of the corresponding annotations
    # NOTE: both svs and geojson files should have the same name in order for the WSIs be mapped correctly to their annotations
    for folder in [hgg_folder, lgg_folder]:
        for file in os.listdir(folder):
            if file.endswith('.svs'):
                wsi_path = os.path.join(folder, file)
                wsi_paths.append(wsi_path)
                label = labels_dict[get_pathology_number(file)]
                labels.append(label)

                # Assuming the annotation files are named as "<wsi_file>.geojson"
                annotation_path = os.path.join(folder, f"{os.path.splitext(file)[0]}.geojson")
                with open(annotation_path, "r") as f:
                    annotation = json.load(f)
                annotations.append(annotation)

    return wsi_paths, labels, annotations

def apply_stain_normalization(patch):
    try:
        patch = np.array(patch)  # Convert the PIL.Image object to a NumPy array
        normalized_patch, _, _ = normalizeStaining(patch)  # Perform H&E stain normalization
        return Image.fromarray(normalized_patch)
    except np.linalg.LinAlgError:
        return patch

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(apply_stain_normalization),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create a folder to store the files that will be used/generated
files_path = 'files'
if files_path is not None:
    os.makedirs(files_path, exist_ok=True)

# Load data and create dataset
wsi_paths, labels, annotations = load_data("PHGG", "LGG", os.path.join(files_path, "Labels.xlsx"))

# Split the data at the WSI level
seed = 42
train_wsi_paths, temp_wsi_paths, train_labels_indices, temp_labels_indices = train_test_split(wsi_paths, np.arange(len(wsi_paths)), test_size=0.3, random_state=seed, stratify=labels)
valid_wsi_paths, test_wsi_paths, valid_labels_indices, test_labels_indices = train_test_split(temp_wsi_paths, np.arange(len(temp_wsi_paths)), test_size=0.5, random_state=seed, stratify=[labels[i] for i in temp_labels_indices])

# Create dictionary mapping from WSI path to label for train, valid and test
train_labels_dict = {path: labels[train_labels_indices[i]] for i, path in enumerate(train_wsi_paths)}
valid_labels_dict = {path: labels[valid_labels_indices[i]] for i, path in enumerate(valid_wsi_paths)}
test_labels_dict = {path: labels[test_labels_indices[i]] for i, path in enumerate(test_wsi_paths)}

# Create dictionary mapping from WSI path to annotations for train, valid and test
train_annotations_dict = {path: annotations[np.where(np.array(wsi_paths) == path)[0][0]] for path in train_wsi_paths}
valid_annotations_dict = {path: annotations[np.where(np.array(wsi_paths) == path)[0][0]] for path in valid_wsi_paths}
test_annotations_dict = {path: annotations[np.where(np.array(wsi_paths) == path)[0][0]] for path in test_wsi_paths}

# Create the training, validation and test datasets
patch_size = 224 # NOTE: Changing the patch size here requires changing the patch_size in the patch classification script
target_magnification = 20 # NOTE: Changing the patch size here requires changing the patch_size in the visualization script
num_patches = 200
# NOTE: changing save_dir requires changing the list of directory names in get_item of PatchDataset
train_dataset = PatchDataset(list(train_labels_dict.keys()), list(train_annotations_dict.values()), target_magnification = target_magnification, base_patch_size=(patch_size, patch_size), num_patches=num_patches, transform=transform, save_dir='train_patches')
valid_dataset = PatchDataset(list(valid_labels_dict.keys()), list(valid_annotations_dict.values()), target_magnification = target_magnification, base_patch_size=(patch_size, patch_size), num_patches=num_patches, transform=transform, save_dir='valid_patches')
test_dataset = PatchDataset(list(test_labels_dict.keys()), list(test_annotations_dict.values()), target_magnification = target_magnification, base_patch_size=(patch_size, patch_size), num_patches=num_patches, transform=transform, save_dir='test_patches')

# Iterate over the datasets to trigger the patch extraction and storing
for idx in range(len(train_dataset)):
    _ = train_dataset[idx]  # The extracted patches and corresponding labels

for idx in range(len(valid_dataset)):
    _ = valid_dataset[idx]  # The extracted patches and corresponding labels

for idx in range(len(test_dataset)):
    _ = test_dataset[idx]  # The extracted patches and corresponding labels

print("Completed patch extraction.")