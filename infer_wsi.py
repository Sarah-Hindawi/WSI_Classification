import os
import time
import pickle
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader
from collections import OrderedDict

import misc
import config
from classification_patches import evaluate
import classification_networks
from patch_dataset import PatchDataset
from directory_patch_dataset import DirectoryPatchDataset

def extract_patches(wsi_paths):

    print('Started patch extraction.')
    start_time = time.time()

    os.makedirs(config.INFER_PATCHES, exist_ok=True)

    dataset = PatchDataset(wsi_paths, annotations=None, max_num_patches=None, coords_file_path=config.COORDS_INFER_FILE_NAME, transform=misc.get_transform(), save_dir=config.INFER_PATCHES)

    # Iterate over the datasets to trigger the patch extraction and storing
    for idx in range(len(dataset)):
        _ = dataset[idx]  # The extracted patches and corresponding labels

    print(f"Completed patch extraction in {time.time() - start_time} seconds.")

def main():

    start_time = time.time()

    # Load WSI paths and labels (annotations are empty)
    wsi_paths = misc.load_paths(config.INFER_WSI_PATH)

    extract_patches(wsi_paths) 

    df = pd.read_excel(config.LABELS_PATH)
    labels_dict = OrderedDict(zip(df['Pathology Number'].str.strip(), df['Label'].str.strip()))

    df = pd.read_excel(config.COORDS_INFER_FILE_NAME)
    coords_dict = OrderedDict(zip(df['patch_id'].str.strip(), zip(df['X'], df['Y'])))

    dataset = DirectoryPatchDataset(config.INFER_PATCHES, labels_dict, coords_dict)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=misc.custom_collate_fn, num_workers=0, pin_memory=True)

    # Load the best model to make inference
    model = classification_networks.ClassificationResNet50()

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    device = misc.get_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    loss, acc, f1, roc_auc, results, features = evaluate(model, dataloader, labels_dict, criterion, device, roc_plot_name='roc_auc_infer_plot.png')
    roc_auc_formatted = f"{float(roc_auc):.3f}" if roc_auc != "N/A" else "N/A"

    print(f"Loss: {loss:.3f}, Accuracy: {acc:.3f}, F1-score: {f1:.3f}, ROC AUC: {roc_auc_formatted}")

    results_df = pd.DataFrame(results)
    results_df.to_excel(config.INFER_PREDICTIONS, index=False)

    pickle.dump(features, open(config.INFER_FEATURES, "wb"))

    print(f"Completed inference in {time.time() - start_time} seconds.")

if __name__ == "__main__":
    main()