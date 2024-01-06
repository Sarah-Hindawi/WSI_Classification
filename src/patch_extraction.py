import os 
import time
import misc
import config
import numpy as np
import pandas as pd
from patch_dataset import PatchDataset
from sklearn.model_selection import train_test_split

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

def extract_patches(wsi_folders=[config.LGG_WSI_PATH], labels_path=config.LABELS_PATH):

    # Load labels from an Excel file
    df = pd.read_excel(labels_path)
    labels_dict = dict(zip(df['Pathology Number'].str.strip(), df['Label'].str.strip()))

    # Check if there are duplicate slides in any of the folders
    if misc.check_duplicate_slides(wsi_folders, case_match=False): # Set case_match to True if we want only one slide per case
        raise Exception('Cannot extract patches, remove duplicate slides and try again.')

    # Load WSI paths, labels, and annotations
    wsi_paths, labels, annotations = misc.load_paths_labels_annotations(wsi_folders, labels_dict)
    wsi_paths_labels = dict(zip(wsi_paths, labels))
    wsi_paths_annotations = dict(zip(wsi_paths, annotations))

    # Convert WSI paths to case IDs
    case_ids = np.array([misc.get_pathology_num_from_labels(path, labels_dict, match_labels = True) for path in wsi_paths])

    # Find unique case IDs and their corresponding labels (needed when there is more than one slide per case)
    unique_case_ids, indices = np.unique(case_ids, return_index=True)
    case_labels = [labels[index] for index in indices]

    # Split the data at the WSI level (train 70%, validation 5%, test 25%)
    seed = 42
    train_case_ids, temp_case_ids, train_case_labels, temp_case_labels = train_test_split(unique_case_ids, case_labels, test_size=0.3, random_state=seed, stratify=case_labels)
    valid_case_ids, test_case_ids, valid_case_labels, test_case_labels = train_test_split(temp_case_ids, temp_case_labels, test_size=0.84, random_state=seed, stratify=temp_case_labels)

    # Gather all WSI paths for each set based on case IDs
    train_wsi_paths = [path for path, case_id in zip(wsi_paths, case_ids) if case_id in train_case_ids]
    valid_wsi_paths = [path for path, case_id in zip(wsi_paths, case_ids) if case_id in valid_case_ids]
    test_wsi_paths = [path for path, case_id in zip(wsi_paths, case_ids) if case_id in test_case_ids]

    # Create dictionary mapping from WSI path to label for train, valid and test
    train_labels_dict = {path: wsi_paths_labels[path] for path in train_wsi_paths}
    valid_labels_dict = {path: wsi_paths_labels[path] for path in valid_wsi_paths}
    test_labels_dict = {path: wsi_paths_labels[path] for path in test_wsi_paths}

    # Create dictionary mapping from WSI path to annotations for train, valid and test
    train_annotations_dict = {path: wsi_paths_annotations[path] for path in train_wsi_paths}
    valid_annotations_dict = {path: wsi_paths_annotations[path] for path in valid_wsi_paths}
    test_annotations_dict = {path: wsi_paths_annotations[path] for path in test_wsi_paths}

    # Create the training, validation and test datasets
    print('Started patch extraction...')
    start_time = time.time()
    transform = misc.get_transform()

    coords_file_path = config.COORDS_FILE_NAME
    max_num_patches = config.MAX_NUM_PATCHES
    train_dataset = PatchDataset(list(train_labels_dict.keys()), list(train_annotations_dict.values()), max_num_patches=max_num_patches, coords_file_path=coords_file_path, transform=transform, save_dir=config.TRAIN_PATCHES, remove_coords=True) 
    valid_dataset = PatchDataset(list(valid_labels_dict.keys()), list(valid_annotations_dict.values()), max_num_patches=max_num_patches, coords_file_path=coords_file_path, transform=transform, save_dir=config.VALID_PATCHES) # accept_blocks=True to allow blocks of the same case
    test_dataset = PatchDataset(list(test_labels_dict.keys()), list(test_annotations_dict.values()), max_num_patches=max_num_patches, coords_file_path=coords_file_path, transform=transform, save_dir=config.TEST_PATCHES)

    # Iterate over the datasets to trigger the patch extraction and storing
    for idx in range(len(train_dataset)):
        _ = train_dataset[idx]  # The extracted patches and corresponding labels

    for idx in range(len(valid_dataset)):
        _ = valid_dataset[idx]  # The extracted patches and corresponding labels

    for idx in range(len(test_dataset)):
        _ = test_dataset[idx]  # The extracted patches and corresponding labels

    print(f"Completed patch extraction for {len(config.CLASSES)} with patch size = {config.PATCH_SIZE} and magnification level = {config.TARGET_MAGNIFICATION} in: {time.time() - start_time} seconds.")

if __name__ == '__main__':
    extract_patches(wsi_folders=[config.LGG_WSI_PATH, config.WSI_BLOCKS_PATH], labels_path=config.LABELS_PATH)