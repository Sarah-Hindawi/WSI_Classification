import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class DirectoryPatchDataset(Dataset):
    def __init__(self, directory, labels, coords, classes, transform=None, augmentation=None):
        self.transform = transform if transform else transforms.ToTensor()
        self.augmentation = augmentation
        self.directory = directory
        self.images = os.listdir(directory)
        self.labels = labels
        self.coords = coords
        self.classes = classes

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
        wsi_idx = list(self.labels.keys()).index(self.get_pathology_num(image_path))
        label = self.get_label(image_path)
        coords = self.get_coords(image_path)

        if self.transform:
            image = self.transform(image)

        # Apply data augmentation to the upsampled images
        if self.augmentation:
            image = self.augmentation(image)

        return image.unsqueeze(0), coords, wsi_idx, label
    
def collate_fn(batch):
    images = []
    coords = []
    wsi_idxs = []
    labels = []

    for image, coord, wsi_idx, label in batch:
        images.append(image)
        coords.append(coord)
        wsi_idxs.append(wsi_idx)
        labels.append(label)

    return images, torch.tensor(coords, dtype=torch.int), torch.tensor(wsi_idxs, dtype=torch.int), torch.tensor(labels, dtype=torch.long)
