import os
import gc
import time
import pickle
import copy
import torch
import misc
import config

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import classification_networks

from collections import OrderedDict
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader 
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, roc_curve, auc
from directory_patch_dataset import DirectoryPatchDataset

torch.manual_seed(42)

class SlideClassifier:

    def __init__(self, model_type='ResNet-50', num_epochs=10, batch_size=8):

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
        torch.autograd.set_detect_anomaly(False)

        self.initialize_model(model_type=model_type)

        df = pd.read_excel(config.LABELS_PATH)
        self.labels_dict = OrderedDict(zip(df['Pathology Number'].str.strip(), df['Label'].str.strip()))

        df = pd.read_excel(config.COORDS_FILE_NAME)
        self.coords_dict = OrderedDict(zip(df['patch_id'].str.strip(), zip(df['X'], df['Y'])))

        train_dataset = DirectoryPatchDataset(config.TRAIN_PATCHES, self.labels_dict, self.coords_dict, augmentation=self.data_augmentation, is_balanced=True)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=misc.custom_collate_fn, num_workers=0, pin_memory=True)

        valid_dataset = DirectoryPatchDataset(config.VALID_PATCHES, self.labels_dict, self.coords_dict)
        self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=misc.custom_collate_fn, num_workers=0, pin_memory=True)

        test_dataset = DirectoryPatchDataset(config.TEST_PATCHES, self.labels_dict, self.coords_dict)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=misc.custom_collate_fn, num_workers=0, pin_memory=True)
        
        # best_model, train_results, train_features = self.train_model_folds(batch_size, num_epochs)
        
        self.train(num_epochs=num_epochs)        


    def train(self, num_epochs=20, early_stopping_patience=3):
        
        best_metric = 0.0
        best_model = None
        best_epoch = None
        early_stopping_counter = 0

        start_time = time.time()

        for epoch in range(1, num_epochs+1):
            print(f"Epoch {epoch}/{num_epochs}")

            running_loss = 0.0
            total_items = 0

            self.model.train()
            for idx, (patches, coords, patch_idxs, wsi_idxs, labels) in enumerate(self.train_loader):
                patches = torch.cat(patches, dim=0).to(self.device)
                labels = labels.clone().detach().to(self.device)
                
                self.optimizer.zero_grad()

                with autocast(): # Enable automatic mixed precision training

                    outputs, features = self.model(patches)

                    loss = self.criterion(outputs, labels)

                    # Accumulate total loss and total items for running loss calculation
                    running_loss += loss.item()
                    total_items += len(patches)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.scheduler.step()

            pred_path = config.add_extension(config.TRAIN_PREDICTIONS, '_epoch' + str(epoch))
            feat_path = config.add_extension(config.TRAIN_FEATURES, '_epoch' +str(epoch))

            train_loss, train_acc, train_f1, train_roc_auc = self.evaluate(pred_path, feat_path, save_model=True)
            print(f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f}, Train ROC AUC: {float(train_roc_auc):.3f}")

            if self.test_loader:
                self.test(epoch)

            if self.valid_loader:
                pred_path = config.add_extension(config.VALID_PREDICTIONS, '_epoch' + str(epoch))
                feat_path = config.add_extension(config.VALID_FEATURES, '_epoch' + str(epoch))

                valid_loss, valid_acc, valid_f1, valid_roc_auc = self.evaluate(pred_path, feat_path)
                print(f"Valid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f}, Valid F1: {valid_f1:.3f}, Valid ROC AUC: {float(valid_roc_auc):.3f}")

                if valid_roc_auc > best_metric:
                    best_model = copy.deepcopy(self.model)
                    best_epoch = epoch
                    best_metric = valid_roc_auc

                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break
            else: 
                best_model = copy.deepcopy(self.model)
            
        print(f"Completed training and saving predictions and feature vectors in: {time.time() - start_time:.2f} seconds.")

        return best_model, best_epoch



    def evaluate(self, results_path, features_path, roc_plot_name=None, save_model=False):
        
        all_labels = []
        all_probs = []
        results = []
        features_results = []

        running_loss = 0.0
        correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for idx, (patches, coords, patch_idxs, wsi_idxs, labels) in enumerate(self.dataloader):
                patches = torch.cat(patches, dim=0).to(self.device)
                labels = labels.clone().detach().to(self.device)

                outputs, features = self.model(patches)
                
                _, predicted = torch.max(outputs, 1)
                probs = torch.softmax(outputs, dim=1)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item() # Add the loss for the current batch of patches (num_patches)

                total += patches.shape[0]
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # Record the model's predictions for each patch
                for i in range(patches.shape[0]):
                    new_row = {
                        # Get pathology number of the WSI that the patch belongs to
                        'wsi_id': list(self.labels_dict.items())[wsi_idxs[i]][0], 
                        'patch_index': patch_idxs[i].item(),
                        'X': coords[i][0].item(),
                        'Y': coords[i][1].item(),
                        'true_label': labels[i].item(),
                        'predicted_label': predicted[i].item(),
                    }
                    
                    for class_index, class_name in enumerate(config.CLASSES):
                        new_row[class_name] = probs[i][class_index].item()
                    
                    results.append(new_row)

                    features_row = {
                        'wsi_id': list(self.labels_dict.items())[wsi_idxs[i]][0],
                        'patch_index': patch_idxs[i].item(),
                        'features': features[i].cpu().numpy()
                    }
                    features_results.append(features_row)

        # Normalize the loss by the total number of patches
        running_loss /= total

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

        self.save_data(self.model, results, features_results, results_path, features_path, save_model)

        return running_loss, correct / total, f1, roc_auc



    def train_model_folds(self, batch_size, num_epochs):
        
        train_dataset = DirectoryPatchDataset(config.TRAIN_PATCHES, self.labels_dict, self.coords_dict, augmentation=self.data_augmentation, is_balanced=True)
        all_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=misc.custom_collate_fn, num_workers=0, pin_memory=True)

        train_labels = [train_dataset.get_label(filename) for filename in train_dataset.images] # Obtain labels for all patches to stratify the split
        
        n_splits = 5
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        best_epochs = []
        
        # NOTE: Split by case ID, not by slide ids
        for fold, (train_idx, valid_idx) in enumerate(kf.split(train_dataset.images, train_labels)):
            print(f"Running Fold {fold + 1}/{n_splits}")

            self.initialize_model() # Reset model, optimizer, scheduler, and scaler

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_idx)

            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler, collate_fn=misc.custom_collate_fn, num_workers=0, pin_memory=True)
            self.valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_subsampler, collate_fn=misc.custom_collate_fn, num_workers=0, pin_memory=True)

            _, best_epoch, _, _, _, _ = self.train(num_epochs)        

            best_epochs.append(best_epoch)

            torch.cuda.empty_cache()
            gc.collect()

        best_epoch = int(np.median(best_epochs))
        print(f"Completed fold training. Best epoch: {best_epoch}. Started training model on all data...")
        
        self.initialize_model()
        
        self.train_loader = all_train_loader
        self.valid_loader = None
    
        best_model, best_epoch, train_results, train_features, _, _= self.train(num_epochs=best_epoch)        

        return best_model, train_results, train_features


    def test(self, epoch):

        pred_path = config.add_extension(config.TEST_PREDICTIONS, '_epoch' + str(epoch))
        feat_path = config.add_extension(config.TEST_FEATURES, '_epoch' + str(epoch))

        test_loss, test_acc, test_f1, test_roc_auc = self.evaluate(pred_path, feat_path, config.ROC_PLOT_PATH)
        print(f"\nTesting the best mode:\nTest Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}, Test F1: {test_f1:.3f}, Test ROC AUC: {float(test_roc_auc):.3f}")

        
    def initialize_model(self, model_type = 'ResNet-50', dropout_rate = config.dropout_rate, weight_decay = 1e-2):

        self.model = classification_networks.ClassificationResNet(model=model_type, dropout_rate=dropout_rate)
        self.criterion = nn.CrossEntropyLoss() 

        self.device = misc.get_device()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
        model.to(self.device)

        self.optimizer = optim.Adam([
        {'params': model.features.parameters(), 'lr': 1e-5, 'weight_decay': weight_decay},  # Lower learning rate for pre-trained layers
        {'params': model.classifier.parameters(), 'lr': 1e-3, 'weight_decay': weight_decay}  # Higher learning rate for the new classifier layer
        ])

        # Set the learning rate scheduler
        self.scheduler = StepLR(self.optimizer, step_size=4, gamma=0.7)
        self.scaler = GradScaler()

        self.data_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomChoice([ 
                transforms.RandomRotation(90),
                transforms.RandomRotation(180),
                transforms.RandomRotation(270),
            ]),
        ])


    def save_data(self, model, results, features, results_path, features_path, save_model=False):

        if save_model: 
            torch.save({'model_state_dict': model.state_dict()}, config.BEST_MODEL_PATH)

        results_df = pd.DataFrame(results)
        results_df.to_excel(results_path, index=False)
        pickle.dump(features, open(features_path, "wb"))
