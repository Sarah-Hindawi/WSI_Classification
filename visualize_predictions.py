import os
import cv2
import gc
import imageio
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import misc
import constants

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

# os.environ['PATH'] = constants.OPENSLIDE_PATH + ";" + os.environ['PATH']
# os.add_dll_directory(constants.OPENSLIDE_PATH)
import openslide

WSI_HEATMAPS_PATH = 'WSI_heatmaps2'


def generate_heatmap(slide_path, patch_classifications, cmap, save_path, level=0, heatmap_level=2):
    slide = openslide.OpenSlide(slide_path)
    patch_size = constants.PATCH_SIZE

    base_magnification = misc.get_base_magnification(slide)
    downsample_factor = base_magnification / constants.TARGET_MAGNIFICATION
    level = slide.get_best_level_for_downsample(downsample_factor)

    if heatmap_level >= len(slide.level_dimensions):
        print('Skipping:', slide_path, 'as the spcified heatmap level doesn\'t exist.')
        return

    heatmap_dims = slide.level_dimensions[heatmap_level]  # Increasing heatmap_level will further downsample the image
    xdim = slide.level_dimensions[level][0] // patch_size
    ydim = slide.level_dimensions[level][1] // patch_size
    downsample = slide.level_downsamples[level]

    pred_arr_hgg = np.full((ydim, xdim), np.nan)
    heatmap_alpha = np.full((ydim, xdim), 0)

    for index, row in patch_classifications.iterrows():
        gc.collect()
        x = round((row['X'] // downsample) / patch_size)
        y = round((row['Y'] // downsample) / patch_size)

        pred_arr_hgg[y, x] = row['HGG']
        heatmap_alpha[y, x] = 128

    # Normalize the prediction arrays between 0 and 1, ignoring NaN values
    pred_arr_hgg = (pred_arr_hgg - np.nanmin(pred_arr_hgg)) / (np.nanmax(pred_arr_hgg) - np.nanmin(pred_arr_hgg))
    # assert pred_arr_hgg.min() >= 0 and pred_arr_hgg.max() <= 1, "pred_arr_hgg has values outside [0, 1]"

    # Show the probabilities of the patches to belong to class 0 (HGG)
    heatmap_hgg = cmap(pred_arr_hgg, bytes=True)  # This is now an RGBA image.
    heatmap_hgg[..., 3] = heatmap_alpha  # Use prediction confidence as alpha channel.
    heatmap_hgg_resized = cv2.resize(heatmap_hgg, heatmap_dims, interpolation=cv2.INTER_NEAREST)

    # Read the whole slide image at the heatmap level
    slide_img = np.array(slide.read_region((0, 0), heatmap_level, heatmap_dims).convert("RGB"))

    # Overlay the heatmap and the slide manually (preserves transparency)
    slide_img = overlay_image(slide_img, heatmap_hgg_resized[:, :, :3], (0, 0), heatmap_hgg_resized[:, :, 3] / 255.0)
    imageio.imwrite(f"{save_path}.jpg", slide_img)

    print(slide_path, 'is saved.')


def overlay_image(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    alpha_mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return img

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    img_combined = np.empty_like(img)
    for c in range(channels):
        img_combined[y1:y2, x1:x2, c] = alpha * img_overlay[y1o:y2o, x1o:x2o, c] + alpha_inv * img[y1:y2, x1:x2, c]

    return img_combined


def process_images(patch_classifications_path, save_dir):
    # Load the classifications dataframe
    patch_classifications_df = pd.read_excel(patch_classifications_path)

    # Group the data by WSI_id
    grouped = patch_classifications_df.groupby('WSI_id')

    cmap = plt.cm.hot
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, orientation='horizontal')
    cbar.set_label('Probability')
    fig.savefig(os.path.join(constants.FILES_PATH, WSI_HEATMAPS_PATH, 'heatmap_legend.png'))

    # For each WSI, create a heatmap
    for wsi_id, group in grouped:

        label = group['True Label'].iloc[0]  # Get the label for the WSI

        if label == 0:
            slide_directory = constants.HGG_PATH
        else:
            slide_directory = constants.LGG_PATH

        # Find slide file based on WSI ID. The WSI name in the predictions.xlsx is only the id of the WSI (e.g. S1-1234, not S1-1234 H&E)
        slide_files = os.listdir(slide_directory)
        slide_filename = next((file for file in slide_files if wsi_id in file and 'svs' in file), None)

        if slide_filename is not None:
            slide_path = os.path.join(slide_directory, slide_filename)
            save_path = os.path.join(save_dir, f"{wsi_id}")
            generate_heatmap(slide_path, group, cmap, save_path)
        else:
            print(f"No slide image file found for WSI ID: {wsi_id}")


def main():
    save_dir = os.path.join(constants.FILES_PATH, WSI_HEATMAPS_PATH)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Call the process_images function
    process_images(constants.TEST_PREDICTIONS, save_dir)


if __name__ == "__main__":
    main()
