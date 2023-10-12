import os
import time
import pickle
import copy
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from collections import Counter
from collections import OrderedDict
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score

import misc
import config 
import classification_networks
from directory_patch_dataset import DirectoryPatchDataset

def train(model, train_loader, valid_loader, labels_dict, criterion, optimizer, scheduler, scaler, device, num_epochs):
    
    best_acc = 0.0
    best_model = None
    early_stopping_counter = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()

        # Initialize running loss
        total_loss = 0.0
        total_items = 0

        # train_loader_iter = tqdm(train_loader, desc="Training", leave=False)

        for idx, (patches, coords, patch_idxs, wsi_idxs, labels) in enumerate(train_loader):
            patches = torch.cat(patches, dim=0).to(device)
            labels = labels.clone().detach().to(device)

            with autocast(): # Enable automatic mixed precision training
                outputs, features = model(patches)

                loss = criterion(outputs, labels)

                # Accumulate total loss and total items for running loss calculation
                total_loss += loss.item()
                total_items += len(patches)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # train_loader_iter.set_postfix({"Loss": loss.item()})

        scheduler.step()

        train_loss, train_acc, train_f1, train_roc_auc, train_results, train_features = evaluate(model, train_loader, labels_dict, criterion, device)
        valid_loss, valid_acc, valid_f1, valid_roc_auc, valid_results, valid_features = evaluate(model, valid_loader, labels_dict, criterion, device)

        if valid_acc > best_acc:
            best_model = copy.deepcopy(model)
            best_acc = valid_acc
            best_train_results = train_results
            best_train_features = train_features
            best_valid_results = valid_results
            best_valid_features = valid_features

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, config.BEST_MODEL_PATH)
            
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        train_roc_auc_formatted = f"{float(train_roc_auc):.3f}" if train_roc_auc != "N/A" else "N/A"
        valid_roc_auc_formatted = f"{float(valid_roc_auc):.3f}" if valid_roc_auc != "N/A" else "N/A"

        # train_loader_iter.close()
        print(f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f}, Train ROC AUC: {train_roc_auc_formatted}")
        print(f"Valid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f}, Valid F1: {valid_f1:.3f}, Valid ROC AUC: {valid_roc_auc_formatted}")

        if early_stopping_counter >= 3:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training time: {time.time() - start_time:.2f} seconds")
    return best_model, best_train_results, best_train_features, best_valid_results, best_valid_features


def evaluate(model, dataloader, labels_dict, criterion, device, roc_plot_name=None):
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    positive_class_probs = []
    results = []
    features_results = []

    running_loss = 0.0

    model.eval()

    with torch.no_grad():
        for idx, (patches, coords, patch_idxs, wsi_idxs, labels) in enumerate(dataloader):
            patches = torch.cat(patches, dim=0).to(device)
            labels = labels.clone().detach().to(device)

            outputs, features = model(patches)
            _, predicted_classes = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            loss = criterion(outputs, labels)

            running_loss += loss.item() # Add the loss for the current batch of patches (num_patches)

            total += patches.shape[0]
            correct += (predicted_classes == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())

            all_probs.extend(probs.cpu().numpy())

            if len(config.CLASSES) == 2:
                positive_class_probs.extend(probs[:, 1].cpu().numpy())

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
        
        if roc_plot_name != None:
            misc.save_roc_auc_plot(fpr, tpr, roc_auc, config.ROC_PLOT_PATH)
        
        roc_auc = sum(roc_auc.values()) / len(roc_auc)

    return running_loss, correct / total, f1, roc_auc, results, features_results



def test(model, best_model, test_loader, labels_dict, criterion, device, roc_plot_name=None):
    # Load the best model and evaluate on the test set
    checkpoint = torch.load(config.BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_f1, test_roc_auc, test_results, test_features = evaluate(best_model, test_loader, labels_dict, criterion, device, roc_plot_name)
    test_roc_auc_formatted = f"{float(test_roc_auc):.3f}" if test_roc_auc != "N/A" else "N/A"

    print(f"\nTesting the best mode:\nTest Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}, Test F1: {test_f1:.3f}, Test ROC AUC: {test_roc_auc_formatted}")

    return test_results, test_features

def train_model(labels=config.LABELS_PATH):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    torch.autograd.set_detect_anomaly(False)  # Disable anomaly detection

    # NOTE: Changing the model (other than ResNet) may require adding transforms.Resize to match the expected input size
    data_augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice([ 
            # Randomly rotate by right angles only
            transforms.RandomRotation(0),
            transforms.RandomRotation(90),
            transforms.RandomRotation(180),
            transforms.RandomRotation(270),
        ]),
    ])

    df = pd.read_excel(labels)
    labels_dict = OrderedDict(zip(df['Pathology Number'].str.strip(), df['Label'].str.strip()))

    df = pd.read_excel(config.COORDS_FILE_NAME)
    coords_dict = OrderedDict(zip(df['patch_id'].str.strip(), zip(df['X'], df['Y'])))

    train_dataset = DirectoryPatchDataset(config.TRAIN_PATCHES, labels_dict, coords_dict, transform=transforms.ToTensor(), augmentation=data_augmentation, is_balanced=True)
    valid_dataset = DirectoryPatchDataset(config.VALID_PATCHES, labels_dict, coords_dict)
    test_dataset = DirectoryPatchDataset(config.TEST_PATCHES, labels_dict, coords_dict)

    # Print the number of occurances of each class in each set 
    train_labels = [train_dataset.get_label(filename) for filename in train_dataset.images]
    valid_labels = [valid_dataset.get_label(filename) for filename in valid_dataset.images]
    test_labels = [test_dataset.get_label(filename) for filename in test_dataset.images]

    train_label_counts = Counter(train_labels)
    valid_label_counts = Counter(valid_labels)
    test_label_counts = Counter(test_labels)

    print("Train dataset class distribution:", train_label_counts)
    print("Validation dataset class distribution:", valid_label_counts)
    print("Test dataset class distribution:", test_label_counts)

    batch_size = 8

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=misc.custom_collate_fn, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=misc.custom_collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=misc.custom_collate_fn, num_workers=0, pin_memory=True)

    dropout_rate = config.dropout_rate
    model_type = 'ResNet-18'
    model = classification_networks.ClassificationResNet(model=model_type, dropout_rate=dropout_rate)

    device = misc.get_device()

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    model.to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    weight = misc.get_class_weights(config.PATCHES_DIR_PATHS, labels_dict, file_type='png', seperator='_')
    weight = weight.to(device) 
    criterion = nn.CrossEntropyLoss(weight=weight)

    # weight_decay = 0
    # if model_type == 'ResNet-50':
    weight_decay = 0.01 # If the model is resnet-50, apply regularization

    optimizer = optim.Adam([
    {'params': model.features.parameters(), 'lr': 1e-5, 'weight_decay': weight_decay},  # Lower learning rate for pre-trained layers
    {'params': model.classifier.parameters(), 'lr': 1e-3, 'weight_decay': weight_decay}  # Higher learning rate for the new classifier layer
    ])

    # Set the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

    scaler = GradScaler()
    
    num_epochs = 15

    # Train the model
    best_model, best_train_results, best_train_features, best_valid_results, best_valid_features = train(model, train_loader, valid_loader, labels_dict, criterion, optimizer, scheduler, scaler, device, num_epochs)
    test_results, test_features = test(model, best_model, test_loader, labels_dict, criterion, device, roc_plot_name='roc_auc_plot.png')

    # Save the best results to Excel files
    train_results_df = pd.DataFrame(best_train_results)
    valid_results_df = pd.DataFrame(best_valid_results)
    test_results_df = pd.DataFrame(test_results)

    train_results_df.to_excel(config.TRAIN_PREDICTIONS, index=False)
    pickle.dump(best_train_features, open(config.TRAIN_FEATURES, "wb"))
    
    valid_results_df.to_excel(config.VALID_PREDICTIONS, index=False)
    pickle.dump(best_valid_features, open(config.VALID_FEATURES, "wb"))

    test_results_df.to_excel(config.TEST_PREDICTIONS, index=False)
    pickle.dump(test_features, open(config.TEST_FEATURES, "wb"))

    print(f"Completed saving all the predictions and feature vectors.")

if __name__ == "__main__":
    train_model(labels=config.LABELS_PATH)