import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score

import misc
import config 
import classification_networks
from directory_patch_dataset import DirectoryPatchDataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
torch.autograd.set_detect_anomaly(False)  # Disable anomaly detection

# Read labels file
df = pd.read_excel(config.LABELS_PATH)
labels_dict = OrderedDict(zip(df['Pathology Number'].str.strip(), df['Label'].str.strip()))

# Read coords file
df = pd.read_excel(os.path.join(config.FILES_PATH, config.COORDS_FILE_NAME))
coords_dict = OrderedDict(zip(df['patch_id'].str.strip(), zip(df['X'], df['Y'])))

# Load datasets from saved patches
test_dataset = DirectoryPatchDataset(config.TEST_PATCHES, labels_dict, coords_dict)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=misc.custom_collate_fn, num_workers=0, pin_memory=True)

# Define the CNN
model = classification_networks.ClassificationResNet18()
criterion = nn.CrossEntropyLoss()

# Set device
device = misc.get_device()

model.to(device)

def evaluate(loader, model, criterion, device, save_roc_auc=False):
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    positive_class_probs = []
    results = []
    features_results = []

    running_loss = 0.0

    with torch.no_grad():
        for idx, (patches, coords, patch_idxs, wsi_idxs, labels) in enumerate(loader):
            patches = torch.cat(patches, dim=0).to(device)
            labels = labels.clone().detach().to(device)

            outputs, features = model(patches)
            _, predicted_classes = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            loss = criterion(outputs, labels)

            # Accumulate loss
            running_loss += loss.item() # Add the loss for the current batch of patches (num_patches)

            total += patches.shape[0]
            correct += (predicted_classes == labels).sum().item()

            # Save labels and predictions for calculation of F1 score and ROC AUC score
            all_labels.extend(labels.cpu().numpy())

            # Saving the softmax probabilities of the positive class to calculate ROC AUC score
            all_probs.extend(probs.cpu().numpy())
            positive_class_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

            # Record the model's predictions for each patch
            for i in range(patches.shape[0]):
                new_row = {
                    'wsi_id': list(labels_dict.items())[wsi_idxs[i]][0], # Get pathology number of the WSI that the patch belongs to
                    'patch_index': patch_idxs[i].item(),
                    'X': coords[i][0].item(),
                    'Y': coords[i][1].item(),
                    'true_label': labels[i].item(),
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

    # Normalize the loss by the total number of patches
    running_loss /= total

    # Calculate F1 score and ROC AUC
    if len(config.CLASSES) == 2:
        threshold = 0.5
        f1 = f1_score(all_labels, np.array(positive_class_probs) > threshold, average='weighted')
        roc_auc = roc_auc_score(all_labels, positive_class_probs)
    else:
        f1 = f1_score(all_labels, np.argmax(all_probs, axis=1), average='weighted')

        # Convert all_labels and all_probs to proper format for multi-class ROC curve computation
        all_labels_bin = label_binarize(all_labels, classes=list(range(len(config.CLASSES))))
        all_probs = np.array(all_probs)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(config.CLASSES)):
            fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        if save_roc_auc:
            misc.save_roc_auc_plot(fpr, tpr, roc_auc, os.path.join(config.FILES_PATH, 'roc_curve512_3.png'))

    return running_loss, correct / total, f1, roc_auc, results, features_results

# Load the best model and evaluate on the test set
checkpoint = torch.load(config.BEST_MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

test_loss, test_acc, test_f1, test_roc_auc, test_results, test_features = evaluate(test_loader, model, criterion, device, save_roc_auc=True)
if len(config.CLASSES) == 2:
    test_roc_auc_formatted = f"{float(test_roc_auc):.3f}" if test_roc_auc != "N/A" else "N/A"
else:
    test_roc_auc_formatted = f"{float(sum(test_roc_auc.values()) / len(test_roc_auc)):.3f}" if test_roc_auc != "N/A" else "N/A"
print(f"\nTesting the best mode:\nTest Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}, Test F1: {test_f1:.3f}, Test ROC AUC: {test_roc_auc_formatted}")