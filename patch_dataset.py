import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from rasterio.features import rasterize
from shapely.geometry import shape
from torch.utils.data import Dataset

import constants
import misc

# os.environ['PATH'] = constants.OPENSLIDE_PATH + ";" + os.environ['PATH']
# os.add_dll_directory(constants.OPENSLIDE_PATH)
import openslide


class PatchDataset(Dataset):
    def __init__(self, wsi_paths, annotations, transform=None, save_dir=None):
        """
        Dataset class for extracting and providing patches from whole-slide images (WSIs).

        Args:
            wsi_paths (list): List of paths to the WSIs.
            annotations (list): List of annotations of ROIs in GeoJSON format corresponding to each WSI.
            transform (callable, optional): Optional transform to be applied to each patch. Defaults to None.
            save_dir (str, optional): Directory to save the extracted patches. Defaults to None.
        """
        self.wsi_paths = wsi_paths
        self.annotations = annotations
        self.transform = transform
        self.save_dir = save_dir  # Save the extracted patches after stain normalization
        self.saved_patches_wsi = set()  # Keep track of the WSIs for which the patches have already been saved
        self.coords_file_path = os.path.join(constants.FILES_PATH, constants.COORDS_FILE_NAME)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # Remove previous coords files if exists
        if os.path.exists(self.coords_file_path):
            os.remove(self.coords_file_path)

    def __len__(self):
        return len(self.wsi_paths)

    def annotation_to_roi_mask(self, annotation, image_dims):
        """Generates a binary mask of the ROI based on a GeoJSON annotation.
        Args:
            annotation (dict): A dictionary containing the GeoJSON annotation.
            image_dims (tuple): The dimensions of the image (width, height).
        Returns:
            np.array: A 2D numpy array representing the binary mask of the ROI.
        """
        shapes = [(shape(feature['geometry']), 1) for feature in annotation['features']]
        # Rasterize the list of shapes onto a binary mask which represent the ROI as a 2D numpy array,
        # where pixels within the ROI have a value of 1 and pixels outside the ROI have a value of 0.
        mask = rasterize(shapes, out_shape=image_dims)
        return mask

    def extract_patches_within_roi(self, wsi_img, roi_mask, best_level, base_patch_size, scaled_patch_size, overlap_percent=20, min_overlap_ratio=0.7):
        """
        Extracts patches from the region of interest (ROI) within a whole-slide image (WSI).

        Args:
            roi_mask (np.array): A binary mask of the ROI, where non-zero values indicate the region of interest.
            best_level (int): The level of the WSI from which to extract the patches. This level provides the desired magnification.
            scaled_patch_size (tuple): The size of the patches to extract, scaled to the target magnification level.
            overlap_percent (int, optional): The desired overlap between patches, specified as a percentage. Defaults to 50.
            min_overlap_ratio (float, optional): The minimum required overlap ratio between a patch and the ROI. Defaults to 0.7 (If at least 70% of the patch lies within the ROI, we accept it).

        Returns:
            list: A list of extracted patches.

        """
        patches_coords = [] # Store a dictionary for each patch, where the key is the patch and the value is its coordiantes

        stride = [int(dim * (1 - overlap_percent / 100)) for dim in scaled_patch_size]  # calculate stride based on the desired overlap
        patch_area = np.prod(scaled_patch_size)
        min_within_roi = patch_area * min_overlap_ratio

        for y in range(0, roi_mask.shape[0], stride[1]):
            for x in range(0, roi_mask.shape[1], stride[0]):
                roi_overlap = np.sum(roi_mask[y:y+scaled_patch_size[1], x:x+scaled_patch_size[0]])
                if roi_overlap >= min_within_roi: # Only extract patches that are mostly insdie the ROI based on min_overlap_ratio
                    coord = (x, y)

                    if best_level < len(wsi_img.level_dimensions):
                        # If the desired level is within the available levels of the WSI
                        patch = np.array(wsi_img.read_region(coord, best_level, scaled_patch_size))[:, :, :3]
                    else:
                        # If the desired level is beyond the available levels of the WSI
                        full_res_patch = np.array(wsi_img.read_region(coord, 0, scaled_patch_size))[:, :, :3]
                        patch = cv2.resize(full_res_patch, scaled_patch_size, interpolation=cv2.INTER_LINEAR)

                    # Ensure that the patch has the same size as base_patch_size
                    if patch.shape[:2] != base_patch_size:
                        # Resize the full-resolution patch to the target size using interpolation
                        patch = cv2.resize(patch, base_patch_size, interpolation=cv2.INTER_LINEAR)

                    h, w, _ = patch.shape
                    if h == base_patch_size[1] and w == base_patch_size[0]:
                        if self.transform:
                            patch = self.transform(patch)
                        patches_coords.append({'patch': patch, 'coord': coord})

                    if len(patches_coords) >= constants.NUM_PATCHES:
                        break
                if len(patches_coords) >= constants.NUM_PATCHES:
                    break
            if len(patches_coords) >= constants.NUM_PATCHES:
                break

        return patches_coords

    # Return patches from a WSI
    def __getitem__(self, index):
        wsi_path = self.wsi_paths[index]
        annotation = self.annotations[index]
        base_patch_size = (constants.PATCH_SIZE, constants.PATCH_SIZE)

        # Get the pathology number from the WSI path
        img_name = os.path.basename(wsi_path)
        pathology_number = misc.get_pathology_number(img_name)

        # Check if patches for the current WSI already exist in any of the sets
        # NOTE: Changing folder names in which the patches will be stored requires changing the list
        for patches_path in [constants.TRAIN_PATH, constants.VALID_PATH, constants.TRAIN_PATH]:
            if os.path.exists(patches_path):
                patch_files = os.listdir(patches_path)
                if any(file.startswith(pathology_number) for file in patch_files):
                    # Patches already exist for this WSI, so skip to the next
                    print('Patches for', pathology_number, 'already exist. Skipped extracting patches.' )
                    return []

        # Load the WSI image
        wsi_img = openslide.OpenSlide(wsi_path)

        # Retrieve the base magnification from the WSI metadata
        base_magnification = misc.get_base_magnification(wsi_img)
        if base_magnification is None:
            print('Base magnification metadata for', pathology_number, 'is missing. Skipped extracting patches.' )
            return []

        # Calculate the patch size for the target magnification
        scale_factor = constants.BASE_MAGNIFICATION / constants.TARGET_MAGNIFICATION
        patch_size = tuple(int(scale_factor * dim) for dim in base_patch_size)

        downsample_factor = base_magnification / constants.TARGET_MAGNIFICATION
        best_level = wsi_img.get_best_level_for_downsample(downsample_factor)

        level_downsample = wsi_img.level_downsamples[best_level]
        scaled_patch_size = tuple(int(np.ceil(ps / level_downsample)) for ps in patch_size)

        # Generate the ROI mask from the annotation
        roi_mask = self.annotation_to_roi_mask(annotation, wsi_img.dimensions[::-1])  # NOTE: dimensions are given in (width, height) but numpy arrays are in (height, width)

        # Extract patches as well as their scaled coordinates from the ROI
        patches_coords = self.extract_patches_within_roi(wsi_img, roi_mask, best_level, base_patch_size, scaled_patch_size, overlap_percent=0, min_overlap_ratio=0.9)

        # Save the normalized patch if a save directory was provided
        if len(patches_coords) > 0 and self.save_dir is not None and wsi_path not in self.saved_patches_wsi:
            self.saved_patches_wsi.add(wsi_path)  # Add the WSI to the set of saved WSIs, only one patch from each WSI is saved

            patches = []
            coords = []
            all_coords = [] # Coordinates for all the patches of all WSIs

            patch_index = 1
            for patch_coords in patches_coords:
                patch = patch_coords['patch']
                coord = patch_coords['coord']
                patch_id = f"{pathology_number}_{patch_index}" # NOTE: Changing patch file name format requires changing extracting patch name in ClassificationWSI.py

                patches.append(patch)
                coords.append(coord)
                all_coords.append({'patch_id': patch_id, 'X': coord[0], 'Y': coord[1]})

                patch_array = patch.numpy().transpose(1, 2, 0)
                # Normalize pixel values to the range of 0-255
                patch_normalized = ((patch_array - patch_array.min()) / (patch_array.max() - patch_array.min())) * 255
                # Convert to uint8
                patch_normalized_uint8 = patch_normalized.astype('uint8')
                patch_pil = Image.fromarray(patch_normalized_uint8)
                patch_pil.save(os.path.join(self.save_dir, patch_id + ".png"))
                patch_index += 1

            # Save coordinates to an Excel file
            new_coords_df = pd.DataFrame(all_coords)

            # If there was already a coords file, it would've been deleted in __init__
            # So if a coords file exists, it would be for previously processed WSIs, so we can safely add to it
            if os.path.exists(self.coords_file_path):
                existing_coords_df = pd.read_excel(self.coords_file_path)
                new_coords_df = pd.concat([existing_coords_df, new_coords_df], ignore_index=True)

            new_coords_df.to_excel(self.coords_file_path, index=False)

            return patches
