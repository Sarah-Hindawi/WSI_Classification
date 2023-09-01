import os
import time
import pickle
import torch
import pandas as pd

from torch.utils.data import DataLoader
from collections import OrderedDict

import misc
import config
import classification_networks
from patch_dataset import PatchDataset
from directory_patch_dataset import DirectoryPatchDataset

def infer_wsi(model, dataloader, labels_dict, device):
    results = []
    features_results = []

    with torch.no_grad():
        for idx, (patches, coords, patch_idxs, wsi_idxs, labels_dict) in enumerate(dataloader):
            patches = torch.cat(patches, dim=0).to(device)
            labels = labels.clone().detach().to(device)

            outputs, features = model(patches)
            _, predicted_classes = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            # Record the model's predictions for each patch
            for i in range(patches.shape[0]):
                new_row = {
                    'wsi_id': list(labels_dict.items())[wsi_idxs[i]][0], # Get pathology number of the WSI that the patch belongs to
                    'patch_index': patch_idxs[i].item(),
                    'X': coords[i][0].item(),
                    'Y': coords[i][1].item(),
                    'predicted_label': predicted_classes[i].item(),
                }

                for class_index, class_name in enumerate(config.CLASSES):
                    new_row[class_name] = probs[i][class_index].item()

                results.append(new_row)

                features_row = {
                    'wsi_id': list(labels_dict.items())[wsi_idxs[i]][0],
                    'patch_index': patch_idxs[i].item(),
                    'features': features[i].cpu().numpy()
                }
                features_results.append(features_row)

    # Save the best results to Excel files
    results_df = pd.DataFrame(results)
    features_results_df = pd.DataFrame(features_results)

    results_df.to_excel(config.INFERENCE_PREDICTIONS, index=False)
    pickle.dump(features_results_df, open(config.INFERENCE_FEATURES, "wb"))

    # return the average probabilities for each class
    return results, features_results


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
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=misc.get_device())

    model = classification_networks.ClassificationResNet50()
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Pass the patches through the trained model and aggregate the patch outputs
    device = misc.get_device()
    model.to(device)

    infer_wsi(model, dataloader, labels_dict, device)

if __name__ == "__main__":
    main()