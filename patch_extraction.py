import os 
import time

import numpy as np
from sklearn.model_selection import train_test_split

import misc
import config
from patch_dataset import PatchDataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

def extract_patches(labels_path=config.LABELS_PATH):

    # Create directories to store extracted patches
    misc.setup_directories()

    # Check if there are duplicate slides in any of the folders
    if misc.check_duplicate_slides(config.WSI_PATHS):
        raise Exception('Cannot extract patches, remove duplicate slides and try again.')

    # Load WSI paths, labels, and annotations
    wsi_paths, labels, annotations = misc.load_paths_labels_annotations(config.WSI_PATHS, labels_path)

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
    print('Started patch extraction.')
    start_time = time.time()
    transform = misc.get_transform()

    # NOTE: Resnet accepts input image size of (224 * 224). Changing patch_size might require adding Resize transform when loading patches before classification
    coords_file_path = config.COORDS_FILE_NAME
    max_num_patches = config.MAX_NUM_PATCHES
    train_dataset = PatchDataset(list(train_labels_dict.keys()), list(train_annotations_dict.values()), max_num_patches=max_num_patches, coords_file_path=coords_file_path, transform=transform, save_dir=config.TRAIN_PATCHES, remove_coords=True)
    valid_dataset = PatchDataset(list(valid_labels_dict.keys()), list(valid_annotations_dict.values()), max_num_patches=max_num_patches, coords_file_path=coords_file_path, transform=transform, save_dir=config.VALID_PATCHES, remove_coords=True)
    test_dataset = PatchDataset(list(test_labels_dict.keys()), list(test_annotations_dict.values()), max_num_patches=max_num_patches, coords_file_path=coords_file_path, transform=transform, save_dir=config.TEST_PATCHES, remove_coords=True)

    # Iterate over the datasets to trigger the patch extraction and storing
    for idx in range(len(train_dataset)):
        _ = train_dataset[idx]  # The extracted patches and corresponding labels

    for idx in range(len(valid_dataset)):
        _ = valid_dataset[idx]  # The extracted patches and corresponding labels

    for idx in range(len(test_dataset)):
        _ = test_dataset[idx]  # The extracted patches and corresponding labels

    print(f"Completed patch extraction for {len(config.CLASSES)} with patch size = {config.PATCH_SIZE} and magnification level = {config.TARGET_MAGNIFICATION} in: {time.time() - start_time} seconds.")

if __name__ == '__main__':
    extract_patches(labels_path=config.LABELS_PATH)