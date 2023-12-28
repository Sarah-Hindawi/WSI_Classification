import os
import time
import misc
import config
import openpyxl
import pandas as pd
from patch_dataset import PatchDataset

def add_patches(wsi_paths, coords_file_path=config.COORDS_FILE_NAME, annotations=None, max_num_patches=None, save_dir=config.TEST_PATCHES, remove_coords = False):
    """Extracts patches from the specified WSIs and saves them in the specified directory. Used for adding paches to the training or test sets."""

    print('Started patch extraction.')
    start_time = time.time()

    os.makedirs(save_dir, exist_ok=True)

    dataset = PatchDataset(wsi_paths, annotations=annotations, max_num_patches=max_num_patches, coords_file_path=coords_file_path, transform=misc.get_transform(), save_dir=save_dir, remove_coords=remove_coords)

    # Iterate over the datasets to trigger the patch extraction and storing
    for idx in range(len(dataset)):
        _ = dataset[idx]  # The extracted patches and corresponding labels

    print(f"Completed patch extraction in {time.time() - start_time} seconds.")


def remove_patches(wsi_file):
    """Remove all patches of the specified WSIs from train, validation, and test sets."""
    
    df = pd.read_excel(wsi_file)
    wsi_ids = df['Pathology Number'].tolist()

    for dataset in [config.TRAIN_PATCHES, config.VALID_PATCHES, config.TEST_PATCHES]:
        patches_in_set = os.listdir(dataset)
        for wsi_id in wsi_ids:
            patches_to_remove = [patch for patch in patches_in_set if patch.startswith(wsi_id)]
            for patch in patches_to_remove:
                os.remove(os.path.join(dataset, patch))
            if (len(patches_to_remove) > 0):
                print(f"Removed patches for {wsi_id} from {dataset}.")

    print("Completed removing patches.")


def move_patches(sheet_path, source_folder, destination_folder, separator=" "):
    """Move files from the source folder to the destination folder based on the file names listed in sheet_path. 
    Important: ALL patches of the same WSI must be moved to the destension folder.
    If moving slides from one folder to another, separator should be space (S30-0000 A1.svs)
    else if moving patches between datasets, separator should be underscore (S30-0000_2.png)"""

    workbook = openpyxl.load_workbook(sheet_path)
    sheet = workbook.active

    # each line in the first column of the excel sheet should correspond to the name of the file to be moved
    files_names = [cell.value for cell in sheet['A']] 

    # Get a list of files in the source folder
    files_in_source_folder = os.listdir(source_folder)

    for file_name in files_in_source_folder:
        if file_name.split(separator)[0] in files_names:
            source_file_path = os.path.join(source_folder, file_name)
            destination_file_path = os.path.join(destination_folder, file_name)
            
            # Move the file to the destination folder
            os.rename(source_file_path, destination_file_path)
            print(f"Moved '{file_name}' to '{destination_folder}'")

    workbook.close()


def find_patches(wsi_id):
    """Find patches of the specified WSIs in either train, validation, and test sets."""

    found_matches = False
    for dataset in [config.TRAIN_PATCHES, config.VALID_PATCHES, config.TEST_PATCHES]:
            patches_in_set = os.listdir(dataset)
            if (len( [patch for patch in patches_in_set if patch.startswith(wsi_id)]) > 0):
                print(f"Found patches for {wsi_id} in {dataset}.")
                found_matches = True

    if not found_matches:
        print(f"No patches found for {wsi_id}.")

    print("Completed searching for patches.")


def get_slides_distribution(datasets):
    """Returns the distribution of the slides in the specified dataset."""

    slides = set()
    patches_in_set = os.listdir(dataset)

    for dataset in datasets:
        for patch in patches_in_set:
            slide = patch.split('_')[0]
            slides.add(slide)

        print("There are", len(slides), "slides in", dataset)

    return slides


def get_slides_stats_from_dir(datasets, labels_file, columns=['Label'], display_results=False):
    """Returns stats of all the slides in the specified dataset directories."""

    df = pd.read_excel(labels_file)
    labels_dict = dict(zip(df['Pathology Number'].str.strip(), df['Label'].str.strip()))

    slides_dict = dict(zip(columns, [[] for _ in range(len(columns))]))
    blocks_dict = dict(zip(columns, [[] for _ in range(len(columns))]))

    blocks = set()

    for dataset in datasets:
        patches_in_set = os.listdir(dataset)

        for patch in patches_in_set:
            slide = misc.get_pathology_num_from_labels(patch.rsplit('_', 1)[0], labels_dict, match_labels=True)
            block = patch.rsplit('_', 1)[0]

            if block not in blocks and slide not in labels_dict.keys():      
                    print(f'Warning: records were not found for {slide}.')
            elif block not in blocks and slide in labels_dict.keys():
                print(f'Slide: {slide}') if display_results else None

                for col in columns:
                    col_value = df[df['Pathology Number'] == slide][col].iloc[0]
                    slides_dict[col].append(col_value)
                    print(f'{col}: {col_value}') if display_results else None
                
            blocks.add(block)

        print("There are", len(blocks), "slides in", dataset)

        # Group the slides by the label and count the frequency of each group
        df = pd.DataFrame(slides_dict)
        df = df.groupby(columns[0]).size().reset_index(name='counts')
        print(df)

        df = pd.DataFrame(blocks_dict)
        df = df.groupby(columns[0]).size().reset_index(name='counts')
        print(df)

        blocks.clear()

    return slides_dict


def get_slides_stats_from_list(slide_list, labels_file, columns=['histology'], display_results=False):
    """Returns stats of the slides in the specified list."""

    df = pd.read_excel(labels_file)
    labels_dict = dict(zip(df['Pathology Number'].str.strip(), df['Label'].str.strip()))

    columns_dict = dict(zip(columns, [[] for _ in range(len(columns))]))

    for slide in set(slide_list):
        if slide not in labels_dict.keys():      
                print(f'Warning: records were not found for {slide}.')
        elif slide in labels_dict.keys():
            print(f'Slide: {slide}') if display_results else None

            for col in columns:
                col_value = df[df['Pathology Number'] == slide][col].iloc[0]
                columns_dict[col].append(col_value)
                print(f'{col}: {col_value}') if display_results else None

    return columns_dict


def main():

    columns = ['Label', 'Pathology Number', 'histology', 'molecular']
    column_values = get_slides_stats_from_dir([config.VALID_PATCHES], config.LABELS_PATH, columns, True)

    plot_col = 'molecular'
    misc.save_stats_plots(column_values[plot_col], config.STATS_PLOT_PATH, title=plot_col, log_scale=False)


if __name__ == '__main__':
    main()