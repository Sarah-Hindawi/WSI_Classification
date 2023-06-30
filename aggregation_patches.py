import os
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def aggregate_predictions(file_path, save_path):

    classifications = pd.read_excel(file_path)

    # Aggregate the classifications by WSI_id
    aggregated_classifications = classifications.groupby('WSI_id').apply(aggregator)

    # Store the predicted labels in a file
    aggregated_classifications.to_excel(save_path, index=True)

    # Calculate and print classification metrices
    true_labels = aggregated_classifications['True Label']
    predicted_labels = aggregated_classifications['Predicted Label']
    predicted_probs = aggregated_classifications['Predicted Probability']

    print('Accuracy:', round(accuracy_score(true_labels, predicted_labels), 2))
    print('F1-score: ', round(f1_score(true_labels, predicted_labels), 2))
    print('ROC AUC: ', round(roc_auc_score(true_labels, predicted_probs), 2))

def aggregator(group):
    # Get all patches for the current WSI
    patches_df = group
    total_patches = len(patches_df)

    # Calculate the number of patches classified as HGG
    hgg_patches = patches_df[(patches_df['HGG'] > patches_df['LGG'])]
    num_hgg_patches = len(hgg_patches)

    # Calculate the number of patches confidently classified as HGG
    hgg_confident_patches = patches_df[(patches_df['HGG'] >= 0.85)]
    num_hgg_confident_patches = len(hgg_confident_patches)

    # Calculate the fraction of patches confidently classified as HGG
    frac_hgg_confident = num_hgg_confident_patches / total_patches

    # If at least 30% of the patches are confidently classified as HGG, then classify the WSI as HGG
    if frac_hgg_confident >= 0.3:
        return pd.Series({'True Label': patches_df['True Label'].iloc[0], 'Predicted Label': 0, 'Predicted Probability': hgg_confident_patches['HGG'].mean()})

    # Check each cell and its surrounding cells for a cluster of at least 5% of the patches classified as HGG
    num_patches = int(total_patches * 0.25)

    if num_hgg_patches > num_patches:

        grid = defaultdict(list)
        patch_size = 224 # NOTE: should match the patch_size defined in PatchExtraction.py
        
        for _, patch in hgg_patches.iterrows():
            # Calculate the grid cell coordinates
            cell_x = patch['X'] // patch_size
            cell_y = patch['Y'] // patch_size
            grid[(cell_x, cell_y)].append(patch)

        for cell, cell_patches in grid.items():
            # Convert the list of patches to a DataFrame
            cluster_patches = pd.DataFrame(cell_patches)
            
            # Check the surrounding cells (2 neighbouring cells in each direction)
            patches_each_direction = num_patches//4
            cells_range = range(-patches_each_direction, patches_each_direction+1)
            for dx in cells_range:
                for dy in cells_range:
                    # Skip the current cell itself
                    if dx == 0 and dy == 0:
                        continue
                    # Add patches from the neighbouring cell to the cluster
                    neighbour_cell = (cell[0] + dx, cell[1] + dy)
                    if neighbour_cell in grid:
                        cluster_patches = pd.concat([cluster_patches, pd.DataFrame(grid[neighbour_cell])], ignore_index=True)

            # Check if we found a cluster of at least 5% of the patches classified as HGG
            if len(cluster_patches) >= num_patches:
                return pd.Series({'True Label': cluster_patches.iloc[0]['True Label'], 'Predicted Label': 0, 'Predicted Probability': cluster_patches['HGG'].mean()})

    # Otherwise, classify the WSI as LGG. Use the average LGG probability for all patches
    lgg_patches = patches_df[(patches_df['LGG'] > patches_df['HGG'])]
    return pd.Series({'True Label': patches_df['True Label'].iloc[0], 'Predicted Label': 1, 'Predicted Probability': lgg_patches['LGG'].mean()})

if __name__ == '__main__':
    
    files_path = 'files'
    
    print("Train set:")
    train_predictions_path = os.path.join(files_path, 'train_predictions_hpf.xlsx')
    train_save_path = os.path.join(files_path, 'train_WSI_classification.xlsx')
    aggregate_predictions(train_predictions_path, train_save_path)

    print("\nValidation set:")
    valid_predictions_path = os.path.join(files_path, 'validation_predictions_hpf.xlsx')
    valid_save_path = os.path.join(files_path, 'valid_WSI_classification.xlsx')
    aggregate_predictions(valid_predictions_path, valid_save_path)

    print("\nTest set:")
    test_predictions_path = os.path.join(files_path, 'test_predictions_hpf.xlsx')
    test_save_path = os.path.join(files_path, 'test_WSI_classification.xlsx')
    aggregate_predictions(test_predictions_path, test_save_path)