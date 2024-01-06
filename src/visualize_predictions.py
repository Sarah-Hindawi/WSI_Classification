import os
import cv2
import misc 
import config
import imageio
import openslide
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

def generate_heatmap(slide_path, patch_classifications, label, cmap, save_path, heatmap_level=1):

    print('Generating heatmaps...')
    
    base_patch_size = (config.PATCH_SIZE, config.PATCH_SIZE)

    wsi_img = openslide.OpenSlide(slide_path)

    base_magnification = misc.get_base_magnification(wsi_img, slide_path)

    downsample_factor = base_magnification / config.TARGET_MAGNIFICATION
    best_level = wsi_img.get_best_level_for_downsample(downsample_factor)

    level_downsample = wsi_img.level_downsamples[best_level]

    patch_size = tuple(int(downsample_factor * dim) for dim in base_patch_size)
    patch_size = tuple(int(np.ceil(ps / level_downsample)) for ps in patch_size)

    if heatmap_level >= len(wsi_img.level_dimensions):
        print('Skipping:', slide_path, 'as the specified heatmap level does not exist.')
        return

    # Increasing heatmap_level will further downsample the image
    heatmap_dims = wsi_img.level_dimensions[heatmap_level] 
    xdim = wsi_img.level_dimensions[best_level][0] // patch_size[0]
    ydim = wsi_img.level_dimensions[best_level][1] // patch_size[1]
    downsample = wsi_img.level_downsamples[best_level]

    pred_arr = np.full((ydim, xdim), np.nan)
    heatmap_alpha = np.full((ydim, xdim), 0)

    for index, row in patch_classifications.iterrows():
        x = int(round((row['X'] / downsample) / patch_size[0]))
        y = int(round((row['Y'] / downsample) / patch_size[1]))

        pred_arr[y, x] = row[label]
        heatmap_alpha[y, x] = 128  # Make the heatmap semitransparent

    # Normalize the prediction arrays between 0 and 1, ignoring NaN values
    pred_arr = (pred_arr - np.nanmin(pred_arr)) / (np.nanmax(pred_arr) - np.nanmin(pred_arr))
    # assert pred_arr.min() >= 0 and pred_arr.max() <= 1, "pred_arr has values outside [0, 1]"

    # Show the probabilities of the patches to belong to the true class
    heatmap = cmap(pred_arr, bytes=True)  # This is now an RGBA image.
    heatmap[..., 3] = heatmap_alpha  # Use prediction confidence as alpha channel.
    heatmap_resized = cv2.resize(heatmap, heatmap_dims, interpolation = cv2.INTER_NEAREST)

    # Read the whole slide image at the heatmap level
    slide_img = np.array(wsi_img.read_region((0, 0), heatmap_level, heatmap_dims).convert("RGB"))

    # Overlay the heatmap and the slide manually (preserves transparency)
    slide_img = overlay_image(slide_img, heatmap_resized[:, :, :3], (0, 0), heatmap_resized[:, :, 3] / 255.0)
    imageio.imwrite(f"{save_path}.png", slide_img)

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
    grouped = patch_classifications_df.groupby('wsi_id')
    
    cmap = plt.cm.hot
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, orientation='horizontal')
    cbar.set_label('Probability')
    fig.savefig(os.path.join(config.FILES_PATH, config.HEATMAPS_PATH, 'heatmap_legend.png'))

    slide_directory = config.LGG_WSI_PATH
    slide_files = os.listdir(slide_directory)

    # For each WSI, create a heatmap
    for wsi_id, group in grouped:
       
        label = config.CLASSES[group['true_label'].iloc[0]]  # Get the label for the WSI

        # Find slide file based on WSI ID. The WSI name in the predictions.xlsx is only the id of the WSI (e.g. S1-1234, not S1-1234 H&E)
        path_nums = [path_num.split()[0] for path_num in slide_files]
   
        if wsi_id not in path_nums: 
            wsi_id = wsi_id[:wsi_id.index('-')+1] + wsi_id[wsi_id.index('-')+2:] # e.g. "S60-0234" => "S60-234"

        slide_filename = next((file for file in slide_files if wsi_id in file.split()[0] and 'svs' in file), None)

        if slide_filename is not None:
            slide_path = os.path.join(slide_directory, slide_filename)
            save_path = os.path.join(save_dir, f"{wsi_id}")
            generate_heatmap(slide_path, group, label, cmap, save_path)
        else:
            print(f"No slide image file found for WSI ID: {wsi_id}")
            
def main():

    save_dir = config.HEATMAPS_PATH

    os.makedirs(save_dir, exist_ok=True)
    process_images(config.TEST_PREDICTIONS, save_dir=save_dir)

if __name__ == "__main__":
    main()
