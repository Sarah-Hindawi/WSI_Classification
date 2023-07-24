import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms

from patch_dataset import PatchDataset
import constants
import misc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"


def load_data(wsi_folders, labels_file):
    wsi_paths = []
    labels = []
    annotations = []

    # Load labels from an Excel file
    df = pd.read_excel(labels_file)
    labels_dict = dict(zip(df['Pathology Number'].str.strip(), df['Label'].str.strip()))

    # Assuming WSIs are divided into two seperate folders (HGG and LGG)
    # Each folder has both svs files of the WSIs, and geojson files of the corresponding annotations
    # NOTE: both svs and geojson files should have the same name in order for the WSIs be mapped correctly to their annotations
    for folder in wsi_folders:
        for file in os.listdir(folder):
            if file.endswith('.svs'):
                wsi_path = os.path.join(folder, file)
                wsi_paths.append(wsi_path)
                label = labels_dict[misc.get_pathology_number(file)]
                labels.append(label)

                # Assuming the annotation files are named as "<wsi_file>.geojson"
                annotation_path = os.path.join(folder, f"{os.path.splitext(file)[0]}.geojson")
                with open(annotation_path, "r") as f:
                    annotation = json.load(f)
                annotations.append(annotation)

    return wsi_paths, labels, annotations

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(misc.apply_stain_normalization),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Standard normalization values since we use pretrained models
])

# Create a folder to store the files that will be used/generated
if constants.FILES_PATH is not None:
    os.makedirs(constants.FILES_PATH, exist_ok=True)

# Load data and create dataset
wsi_paths, labels, annotations = load_data(constants.WSI_PATHS, constants.LABELS_PATH)

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
# NOTE: Changing the patch size here requires changing the patch_size in the patch classification script
# NOTE: Resnet accepts input image size of (224 * 224). Changing patch_size will require adding Resize transform when loading patches before classification
# NOTE: Changing save_dir requires changing the list of directory names in get_item of PatchDataset
# NOTE: When attempting to re-extract all the patches, delete the train, validation and test patches as WSIs with patches in any of these folders will not be extracted again
train_dataset = PatchDataset(list(train_labels_dict.keys()), list(train_annotations_dict.values()), transform=transform, save_dir=constants.TRAIN_PATH)
valid_dataset = PatchDataset(list(valid_labels_dict.keys()), list(valid_annotations_dict.values()), transform=transform, save_dir=constants.VALID_PATH)
test_dataset = PatchDataset(list(test_labels_dict.keys()), list(test_annotations_dict.values()), transform=transform, save_dir=constants.TEST_PATH)

# Iterate over the datasets to trigger the patch extraction and storing
for idx in range(len(train_dataset)):
    _ = train_dataset[idx]  # The extracted patches and corresponding labels

for idx in range(len(valid_dataset)):
    _ = valid_dataset[idx]  # The extracted patches and corresponding labels

for idx in range(len(test_dataset)):
    _ = test_dataset[idx]  # The extracted patches and corresponding labels

print("Completed patch extraction in:", str(time.time() - start_time), 'seconds')