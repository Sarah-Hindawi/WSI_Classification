import copy
import os
import torch
import time
import pickle
import platform
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import transforms
from collections import OrderedDict
from torch.optim.lr_scheduler import StepLR
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, roc_auc_score

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
torch.autograd.set_detect_anomaly(False)  # Disable anomaly detection

class ClassificationNetwork(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.1):
        super(ClassificationNetwork, self).__init__()
        # Load the pretrained resnet18 model
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Retain all layers of the original model, only change the final layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_features = resnet.fc.in_features

        # Add a new fully connected layer for classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)  # Binary classification
        )

        # Initialize weights of the new classifier layers
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        outputs = self.classifier(features)
        return outputs, features

class DirectoryPatchDataset(Dataset):
    def __init__(self, directory, labels, coords, classes, transform=None, augmentation=None):
        self.directory = directory
        self.images = os.listdir(directory)
        self.labels = labels
        self.coords = coords
        self.classes = classes
        self.transform = transform if transform else transforms.ToTensor()
        self.augmentation = augmentation

        # Initialize image lists for each class
        self.image_list = {class_name: [] for class_name in self.classes}

        # Populate image lists
        for img in self.images:
            if img.endswith('png'):
                label = self.get_label(img)
                self.image_list[self.classes[label]].append(img)

        # Identify the majority class
        majority_class = self.find_majority_class()

        # Upsample all classes to match the majority class
        for class_name, images in self.image_list.items():
            if class_name != majority_class:
                count_diff = len(self.image_list[majority_class]) - len(images)
                upsampling_indices = np.random.choice(len(images), size=count_diff, replace=True)
                upsampling_images = [images[i] for i in upsampling_indices]
                images.extend(upsampling_images)

        # Flatten the lists for easy access in __getitem__
        self.images = [image for images in self.image_list.values() for image in images]

    def find_majority_class(self):
        class_counts = {class_name: len(images) for class_name, images in self.image_list.items()}
        majority_class = max(class_counts, key=class_counts.get)
        return majority_class

    def get_pathology_num(self, file_path):
        # Extract pathology number from filename which appears before an underscore (as defined in PatchExtraction.py)
        # Note: Changing patch image file name format in PatchExtraction.py requires changing this line
        filename = os.path.basename(file_path)
        wsi_id = filename.split("_")[0]
        return wsi_id

    def get_patch_index(self, file_path):
        filename = os.path.basename(file_path)
        patch_idx = int(os.path.splitext(filename.split("_")[1])[0])
        return patch_idx

    def get_label(self, filename):
        # Get label from labels DataFrame
        return self.classes.index(self.labels[self.get_pathology_num(filename)])

    def get_coords(self, file_path):
        filename = os.path.basename(file_path)
        patch_id = filename.split(".")[0]
        return [self.coords[patch_id][0], self.coords[patch_id][1]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.directory, self.images[idx])
        image = Image.open(image_path).convert('RGB')
        patch_idx = self.get_patch_index(image_path)
        wsi_idx = list(self.labels.keys()).index(self.get_pathology_num(image_path))
        label = self.get_label(image_path)
        coords = self.get_coords(image_path)

        if self.transform:
            image = self.transform(image)

        # Apply data augmentation to the upsampled images
        if self.augmentation:
            image = self.augmentation(image)

        return image.unsqueeze(0), coords, patch_idx, wsi_idx, label

def collate_fn(batch):
    images = []
    coords = []
    patch_idxs = []
    wsi_idxs = []
    labels = []

    for image, coord, patch_idx, wsi_idx, label in batch:
        images.append(image)
        coords.append(coord)
        patch_idxs.append(patch_idx)
        wsi_idxs.append(wsi_idx)
        labels.append(label)

    return images, torch.tensor(coords, dtype=torch.int), torch.tensor(patch_idxs, dtype=torch.int), torch.tensor(wsi_idxs, dtype=torch.int), torch.tensor(labels, dtype=torch.long)

# Define a series of data augmentations
# NOTE: Changing the model (other than ResNet) will require adding transforms.Resize to match the expected input size
data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
])

# Read labels file
files_path = 'files'
df = pd.read_excel(os.path.join(files_path, "Labels.xlsx"))
labels_dict = OrderedDict(zip(df['Pathology Number'].str.strip(), df['Label'].str.strip()))

# Read coords file
df = pd.read_excel(os.path.join(files_path, "patches_coords.xlsx"))
coords_dict = OrderedDict(zip(df['patch_id'].str.strip(), zip(df['X'], df['Y'])))

# Note: Changing patch image file name format in PatchExtraction.py requires changing this line
classes = ['HGG', 'LGG']

# Load datasets from saved patches
train_dataset = DirectoryPatchDataset('train_patches', labels_dict, coords_dict, classes=classes, transform=transforms.ToTensor(), augmentation=data_augmentation)
valid_dataset = DirectoryPatchDataset('valid_patches', labels_dict, coords_dict, classes=classes)
test_dataset = DirectoryPatchDataset('test_patches', labels_dict, coords_dict, classes=classes)

# Create DataLoaders for training and test datasets
batch_size = 6

if platform.system() == "Windows":
    num_workers = 0
else:
    num_workers = torch.get_num_threads()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

# Define the CNN
model = ClassificationNetwork()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Check if multiple GPUs are available and wrap model in nn.DataParallel if they are
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to(device)

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set the learning rate scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

scaler = GradScaler()

# Train the model
num_epochs = 5

def evaluate(loader, model, criterion, device):
    correct = 0
    total = 0
    all_labels = []
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

            loss = criterion(outputs, labels)

            # Accumulate loss
            running_loss += loss.item() # Add the loss for the current batch of patches (num_patches)

            total += patches.shape[0]
            correct += (predicted_classes == labels).sum().item()

            # Save labels and predictions for calculation of F1 score and ROC AUC score
            all_labels.extend(labels.cpu().numpy())

            # Saving the softmax probabilities of the positive class to calculate ROC AUC score
            positive_class_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

            # Record the model's predictions for each patch
            probs = torch.softmax(outputs, dim=1)
            for i in range(patches.shape[0]):
                new_row = {
                    'WSI_id': list(labels_dict.items())[wsi_idxs[i]][0], # Get pathology number of the WSI that the patch belongs to
                    'Patch_index': patch_idxs[i].item(),
                    'X': coords[i][0].item(),
                    'Y': coords[i][1].item(),
                    classes[0]: probs[i][0].item(),
                    classes[1]: probs[i][1].item(),
                    'True Label': labels[i].item(),
                    'Predicted Label': predicted_classes[i].item(),
                }
                results.append(new_row)

                features_row = {
                    'WSI_id': list(labels_dict.items())[wsi_idxs[i]][0],
                    'Patch_index': patch_idxs[i].item(),
                    'Features': features[i].cpu().numpy()
                }
                features_results.append(features_row)

    # Normalize the loss by the total number of patches
    running_loss /= total

    # Calculate F1 score and ROC AUC score
    Threshold = 0.5
    f1 = f1_score(all_labels, np.array(positive_class_probs) > Threshold, average='weighted')

    if len(np.unique(all_labels)) > 1:
        roc_auc = roc_auc_score(all_labels, positive_class_probs)
    else:
        roc_auc = 'N/A'

    return running_loss, correct / total, f1, roc_auc, results, features_results

# Initialize the best accuracy variable and model path
best_acc = 0.0
best_model = None
model_save_path = os.path.join(files_path, "best_model.pth")

# Training loop
start_time = time.time()

for epoch in range(num_epochs):
    num_classes = 2

    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()

    # Initialize running loss
    total_loss = 0.0
    total_items = 0

    # Training loop
    for idx, (patches, coords, patch_idxs, wsi_idxs, labels) in enumerate(train_loader):
        patches = torch.cat(patches, dim=0).to(device)
        labels = labels.clone().detach().to(device)

        optimizer.zero_grad()

        # Forward pass
        with autocast(): # Enable automatic mixed precision training
            outputs, features = model(patches)

            loss = criterion(outputs, labels)

            # Accumulate total loss and total items for running loss calculation
            total_loss += loss.item()
            total_items += len(patches)

        # Backward pass and optimization step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Compute average loss over the entire epoch
    running_loss = total_loss / total_items if total_items > 0 else 0.0

    # Step the learning rate scheduler
    scheduler.step()

    # Calculate loss, accuracy, F1 score, and ROC AUC scores
    model.eval()

    train_loss, train_acc, train_f1, train_roc_auc, train_results, train_features = evaluate(train_loader, model, criterion, device)
    valid_loss, valid_acc, valid_f1, valid_roc_auc, valid_results, valid_features = evaluate(valid_loader, model, criterion, device)

    # Save the model if it has the best validation accuracy so far
    if valid_acc > best_acc:
        best_model = copy.deepcopy(model)  # save the best model
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
        }, model_save_path)

    train_roc_auc_formatted = f"{float(train_roc_auc):.3f}" if train_roc_auc != "N/A" else "N/A"
    valid_roc_auc_formatted = f"{float(valid_roc_auc):.3f}" if valid_roc_auc != "N/A" else "N/A"

    print(f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f}, Train ROC AUC: {train_roc_auc_formatted}")
    print(f"Valid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f}, Valid F1: {valid_f1:.3f}, Valid ROC AUC: {valid_roc_auc_formatted}")

# Load the best model and evaluate on the test set
checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.eval()

test_loss, test_acc, test_f1, test_roc_auc, test_results, test_features = evaluate(test_loader, best_model, criterion, device)
test_roc_auc_formatted = f"{float(test_roc_auc):.3f}" if test_roc_auc != "N/A" else "N/A"
print(f"\nTesting the best mode:\nTest Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}, Test F1: {test_f1:.3f}, Test ROC AUC: {test_roc_auc_formatted}")

# Save the best results to Excel files
train_results_df = pd.DataFrame(best_train_results)
valid_results_df = pd.DataFrame(best_valid_results)
test_results_df = pd.DataFrame(test_results)

files_path = 'files'

train_results_df.to_excel(os.path.join(files_path, 'train_predictions.xlsx'))
pickle.dump(best_train_features, open(os.path.join(files_path, "train_features.pkl"), "wb" ))

valid_results_df.to_excel(os.path.join(files_path, 'validation_predictions.xlsx'))
pickle.dump(best_valid_features, open(os.path.join(files_path, "valid_features.pkl"), "wb" ))

test_results_df.to_excel(os.path.join(files_path, 'test_predictions.xlsx'))
pickle.dump(test_features, open(os.path.join(files_path, "test_features.pkl"), "wb" ))

print("Training complete in:", str(time.time() - start_time))
