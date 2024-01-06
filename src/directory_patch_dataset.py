import os
import config
import misc 
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class DirectoryPatchDataset(Dataset):
    def __init__(self, directory, labels, coords, wsi_filenames, transform=None, augmentation=None, is_balanced = False):
        self.directory = directory
        self.images = os.listdir(directory)
        self.labels = labels
        self.coords = coords
        self.wsi_filenames = wsi_filenames
        self.augmentation = augmentation
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        # Initialize image lists for each class
        self.image_list = {class_name: [] for class_name in config.CLASSES}

        # Populate image lists
        for img in self.images:
            if self.labels and img.endswith('png'):
                pathology_num = misc.get_pathology_num_from_labels(img, self.labels, match_labels=True)
                label = self.get_label(pathology_num)
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

    def get_patch_index(self, file_path):
        filename = os.path.basename(file_path)
        patch_idx = int(filename.split(".")[-2][-1]) # S80-1234 A2_3.png' => get 3
        return patch_idx

    def get_label(self, filename):
        pathology_num = misc.get_pathology_num_from_labels(filename, self.labels, match_labels=True, separator="_")
        # Get label from labels DataFrame
        if self.labels and pathology_num in self.labels.keys():
            return config.CLASSES.index(self.labels[pathology_num])
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
        img_name = os.path.basename(image_path).rsplit('_', 1)[0] + '.svs'
        image = Image.open(image_path).convert('RGB')
        patch_idx = self.get_patch_index(image_path)
        coords = self.get_coords(image_path)

        pathology_num = misc.get_pathology_num_from_labels(image_path, self.labels, match_labels=True, separator="_")

        if self.wsi_filenames and img_name in self.wsi_filenames:
            wsi_idx = self.wsi_filenames.index(img_name)
        else: 
            wsi_idx = None
            print(f'Could not find {img_name} in {config.WSI_FILENAMES}. Skipping...')

        if self.labels and pathology_num in self.labels.keys():
            label = self.get_label(pathology_num)
        else:
            label = None    
            print(f'Could not find the label for slide: {img_name}. Skipping...')

        if self.transform:
            image = self.transform(image)

        # Apply data augmentation to the upsampled images
        if self.augmentation:
            image = self.augmentation(image)

        return image.unsqueeze(0), coords, patch_idx, wsi_idx, label
