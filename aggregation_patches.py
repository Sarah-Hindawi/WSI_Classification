import os
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score

import config as c

def aggregate_predictions(preds_path, save_path):

    preds = pd.read_excel(preds_path)

    # Aggregate the classifications by WSI_id
    aggregated_classifications = preds.groupby('wsi_id').apply(aggregator)

    # Store the predicted labels in a file
    aggregated_classifications.to_excel(save_path, index=True)

    # Calculate and print classification metrices
    true_labels = aggregated_classifications['True Label']
    predicted_labels = aggregated_classifications['Predicted Label']
    predicted_probs = aggregated_classifications['Predicted Probability']

    print('Accuracy:', round(accuracy_score(true_labels, predicted_labels), 2))
    print('ROC AUC: ', round(roc_auc_score(true_labels, predicted_probs), 2))

def aggregator(group):
    # Get all patches for the current WSI
    patches_df = group
    total_patches = len(patches_df)

    # Calculate the number of patches classified as HGG
    hgg_patches = patches_df[(patches_df['HGG'] > patches_df['LGG'])]
    num_hgg_patches = len(hgg_patches)

    # Calculate the number of patches confidently classified as HGG
    confidence_threshold = 0.85
    hgg_confident_patches = patches_df[(patches_df['HGG'] >= confidence_threshold)]
    num_hgg_confident_patches = len(hgg_confident_patches)

    # Calculate the fraction of patches confidently classified as HGG
    frac_hgg_confident = num_hgg_confident_patches / total_patches

    # If at least 30% of the patches are confidently classified as HGG, then classify the WSI as HGG
    if frac_hgg_confident >= 0.3:
        return pd.Series({'True Label': patches_df['true_label'].iloc[0], 'Predicted Label': 0, 'Predicted Probability': hgg_confident_patches['HGG'].mean()})

    # Check each cell and its surrounding cells for a cluster of at least 5% of the total patches that are classified as HGG
    min_patches = int(total_patches * 0.25)

    if num_hgg_patches >= min_patches:

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

            # Check the surrounding cells
            patches_each_direction = min_patches//4
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
            if len(cluster_patches) >= min_patches:
                return pd.Series({'True Label': cluster_patches.iloc[0]['true_label'], 'Predicted Label': 0, 'Predicted Probability': cluster_patches['HGG'].mean()})

    # Otherwise, classify the WSI as LGG. Use the average LGG probability for all patches
    lgg_patches = patches_df[(patches_df['LGG'] > patches_df['HGG'])]
    return pd.Series({'True Label': patches_df['true_label'].iloc[0], 'Predicted Label': 1, 'Predicted Probability': lgg_patches['LGG'].mean()})

# def aggregate_predictions_ML(train_predictions_path, train_features_path, train_save_path, valid_predictions_path, valid_features_path,
#                              valid_save_path, test_predictions_path, test_features_path, test_save_path, save_path, method = "SVM"):

#     train_preds = pd.read_excel(train_predictions_path)
#     train_features = pd.DataFrame(pd.read_pickle(train_features_path))

#     valid_preds = pd.read_excel(valid_predictions_path)
#     valid_features = pd.DataFrame(pd.read_pickle(valid_features_path))

#     test_preds = pd.read_excel(test_predictions_path)
#     test_features = pd.DataFrame(pd.read_pickle(test_features_path))

#     train_patches_df = train_preds.merge(train_features, on=["wsi_id", "patch_index"])
#     valid_patches_df = valid_preds.merge(valid_features, on=["wsi_id", "patch_index"])
#     test_patches_df = test_preds.merge(test_features, on=["wsi_id", "patch_index"])

#     # Aggregate the classifications by WSI_id
#     aggregated_classifications = SVM_aggregator(train_patches_df, valid_patches_df, test_patches_df)

#     # Store the predicted labels in a file
#     aggregated_classifications.to_excel(save_path, index=True)

#     # Calculate and print classification metrics
#     true_labels = aggregated_classifications['True Label']
#     predicted_labels = aggregated_classifications['Predicted Label']
#     predicted_probs = aggregated_classifications['Predicted Probability']

#     print('Accuracy:', round(accuracy_score(true_labels, predicted_labels), 2))
#     print('ROC AUC: ', round(roc_auc_score(true_labels, predicted_probs), 2))

# def SVM_aggregator(train_patches_df, valid_patches_df, test_patches_df):
    # for WSI in train_patches_df.groupby('WSI_id'):
    #
    # # Get all patches for the current WSI
    # patches_features = patches_df['Features'] # Feature vector extracted from CNN on each patch
    #
    # clf = SVC(probability=True)
    #
    # # CODE to extract
    #
    #
    # # Train SVM model on feature vectors extracted from the CNN previously
    # clf.fit(feature_vectors['train'], labels['train'])
    #
    # # evaluate on test feature vectors and output predictions
    # preds = clf.predict_proba(self.fvs_dict['test'])
    #
    # self.predictions_dict['test'] = preds[:,1]
    # return self.predictions_dict, self.labels_dict

if __name__ == '__main__':

    files_path = c.FILES_PATH

    print("Train set:")
    train_predictions_path = os.path.join(files_path, c.TRAIN_PREDICTIONS)
    train_features_path = os.path.join(files_path, c.TRAIN_FEATURES)
    train_save_path = os.path.join(files_path, c.TRAIN_WSI_CLASSIFICATION)
    aggregate_predictions(train_predictions_path, train_save_path)

    print("\nValidation set:")
    valid_predictions_path = os.path.join(files_path, c.VALID_PREDICTIONS)
    valid_features_path = os.path.join(files_path, c.VALID_FEATURES)
    valid_save_path = os.path.join(files_path, c.VALID_WSI_CLASSIFICATION)
    aggregate_predictions(valid_predictions_path, valid_save_path)

    print("\nTest set:")
    test_predictions_path = os.path.join(files_path, c.TEST_PREDICTIONS)
    test_features_path = os.path.join(files_path, c.TEST_FEATURES)
    test_save_path = os.path.join(files_path, c.TEST_WSI_CLASSIFICATION)
    aggregate_predictions(test_predictions_path, test_save_path)

    # aggregate_predictions_ML(train_predictions_path, train_features_path, train_save_path,
    #                          valid_predictions_path, valid_features_path, valid_save_path,
    #                          test_predictions_path, test_features_path, test_save_path, "SVM")
