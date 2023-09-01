import os
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

import config 

class DirectoryPatchDataset(Dataset):
    def __init__(self, directory, labels, coords, transform=None, augmentation=None, is_balanced = False):
        self.directory = directory
        self.images = os.listdir(directory)
        self.labels = labels
        self.coords = coords
        self.transform = transform if transform else transforms.ToTensor()
        self.augmentation = augmentation

        # Initialize image lists for each class
        self.image_list = {class_name: [] for class_name in config.CLASSES}

        # Populate image lists
        for img in self.images:
            if self.labels and img.endswith('png'):
                label = self.get_label(img)
                self.image_list[config.CLASSES[label]].append(img)

        # If the dataset should be balanced for training purposes 
        if is_balanced: 
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
        if self.labels and self.get_pathology_num(filename) in self.labels.keys():
            return config.CLASSES.index(self.labels[self.get_pathology_num(filename)])
        else:
            return None

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
        coords = self.get_coords(image_path)

        if self.labels and self.get_pathology_num(image_path) in self.labels.keys():
            wsi_idx = list(self.labels.keys()).index(self.get_pathology_num(image_path))
            label = self.get_label(image_path)
        else:
            wsi_idx = label = None    

        if self.transform:
            image = self.transform(image)

        # Apply data augmentation to the upsampled images
        if self.augmentation:
            image = self.augmentation(image)

        return image.unsqueeze(0), coords, patch_idx, wsi_idx, label
