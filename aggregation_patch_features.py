import misc
import math
import pickle
import numpy as np
import pandas as pd
import config as c
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Approach 1: Average all the feature vectors of the patches in a slide. 
# Provides a general representation of the slide, but might overlook key regions
def pool_all_features(patches_df, save_path, method='avg'):

    aggregated_data = []

    for wsi_id, group in patches_df.groupby('wsi_id'):
        if method == 'avg':
            wsi_features = np.mean(np.vstack(group['features'].values).reshape(-1,c.FEATURE_VECTOR_SIZE), axis=0).reshape(-1,1).T
        elif method == 'max':
            wsi_features = np.max(np.vstack(group['features'].values).reshape(-1, c.FEATURE_VECTOR_SIZE), axis=0).reshape(-1,1).T

        row = {
            'wsi_id': wsi_id,
            'label': group.iloc[0]['true_label'],
            'features': wsi_features
        }
        aggregated_data.append(row)

    aggregated_data_df = pd.DataFrame(aggregated_data)
    pickle.dump(aggregated_data_df, open(save_path, "wb"))  # Save the aggregated features

    return aggregated_data_df

# Approach 2: Average top n the feature vectors of the patches in a slide. 
def pool_top_features(patches_df, save_path, method='avg', min_patches=1, top_percent=0.5, min_top_patches=10):
    """Aggregate the feature vectors of the top k patches per slide based on their confidence scores.
    Args:
        patches_df: DataFrame containing the feature vectors of all patches
        save_path: Path to save the aggregated features to
        method: Method to aggregate the features. Either 'avg' or 'max'
        min_patches: Minimum number of patches required per slide to be used in the aggregation
        top_percent: Percentage of top patches to aggregate per slide
        min_top_patches: Minimum number of top patches to aggregate per slide

    Returns:
        aggregated_data_df: DataFrame containing the aggregated a single feature vector per slide
    """

    # Calculate maximum confidence for each patch
    patches_df['max_confidence'] = patches_df[c.CLASSES].max(axis=1)
    
    # Remove duplicates: keep the highest confidence patch for patches with same patch_index & wsi_id (data augmentation)
    patches_df = patches_df.sort_values('max_confidence', ascending=False)
    patches_df = patches_df.drop_duplicates(subset=['wsi_id', 'patch_index'])

    aggregated_data = []

    for wsi_id, group in patches_df.groupby('wsi_id'):

        if len(group) < min_patches:
            print(f"Slide {wsi_id} has only {len(group)} patches. Skipping...")
            continue

        top_patches = int(len(group) * top_percent)
        top_patches = min_top_patches if top_patches < min_top_patches else top_patches  # Ensure at least 10 patches are selected

        # Get the top k patches for this WSI based on max confidence
        top_patches_group = group.nlargest(top_patches, 'max_confidence')

        # Aggregate the features of these top patches
        features_array = np.vstack(top_patches_group['features'].values).reshape(-1, c.FEATURE_VECTOR_SIZE)

        if method == 'avg':
            wsi_features = np.mean(features_array, axis=0).reshape(-1,1).T
        elif method == 'max':
            wsi_features = np.max(features_array, axis=0).reshape(-1,1).T

        row = {
            'wsi_id': wsi_id,
            'label': top_patches_group.iloc[0]['true_label'],
            'features': wsi_features
        }
        aggregated_data.append(row)

    # Create DataFrame with the aggregated slide-level feature vectors
    aggregated_data_df = pd.DataFrame(aggregated_data)
    
    # Save the aggregated features to a file
    pickle.dump(aggregated_data_df, open(save_path, "wb"))

    return aggregated_data_df

def plot_features(patches_df, save_path, num_patches=100):
    """Plot num_patches patches/feature vectors with the highest confidence and 
            num_patches with the lowest confidence for each slide."""
   
    # Calculate maximum confidence for each patch
    patches_df['max_confidence'] = patches_df[c.CLASSES].max(axis=1)
    
    # Remove duplicates: keep the highest confidence patch for patches with same patch_index & wsi_id
    patches_df = patches_df.sort_values('max_confidence', ascending=False)
    patches_df = patches_df.drop_duplicates(subset=['wsi_id', 'patch_index'])

    for wsi_id, group in patches_df.groupby('wsi_id'):
        highest_confidence_patches = group.nlargest(num_patches, 'max_confidence')['features']
        lowest_confidence_patches = group.nsmallest(num_patches, 'max_confidence')['features']

        # Ensure all features are numpy arrays and have the same length
        highest_confidence_features = np.array([np.array(feature) for feature in highest_confidence_patches if isinstance(feature, np.ndarray)])
        lowest_confidence_features = np.array([np.array(feature) for feature in lowest_confidence_patches if isinstance(feature, np.ndarray)])

        if highest_confidence_features.size == 0 or lowest_confidence_features.size == 0:
            print(f"No valid features found for slide {wsi_id}")
            continue

        # Perform PCA to reduce to two dimensions for plotting
        pca = PCA(n_components=2)
        combined_features = np.vstack((highest_confidence_features, lowest_confidence_features))
        reduced_features = pca.fit_transform(combined_features)

        # Split back into highest and lowest confidence features
        highest_confidence_reduced = reduced_features[:num_patches, :]
        lowest_confidence_reduced = reduced_features[num_patches:, :]

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.scatter(highest_confidence_reduced[:, 0], highest_confidence_reduced[:, 1], color='blue', label='Highest Confidence Patches')
        plt.scatter(lowest_confidence_reduced[:, 0], lowest_confidence_reduced[:, 1], color='red', label='Lowest Confidence Patches')
        plt.title(f'PCA-Reduced Feature Space for {wsi_id}')
        plt.xlabel('PCA Feature 1')
        plt.ylabel('PCA Feature 2')
        plt.legend()
        plt.savefig(f'{save_path}_feature_vectors_{wsi_id}.png')
        plt.close()


# Approach 3: Pool features spatially using a fixed-size grid.
def pool_features_spatial(patches_df, save_path, method='avg'):
    aggregated_data = []
    
    # Define a fixed-size grid based on the largest slide dimensions 
    GRID_SIZE = math.ceil(misc.get_largest_slide_dimensions(patches_df) / c.PATCH_SIZE)
    
    for wsi_id, group in patches_df.groupby('wsi_id'):
        # Initialize an empty feature grid filled with NaNs to represent absent values
        feature_grid = np.full((GRID_SIZE, GRID_SIZE, c.FEATURE_VECTOR_SIZE), np.nan)
        
        for _, row in group.iterrows():
            # Calculate grid coordinates for each patch based on its top-left corner (X, Y)
            grid_x = int(row['X'] / 512) % GRID_SIZE
            grid_y = int(row['Y'] / 512) % GRID_SIZE
            feature_grid[grid_x, grid_y] = row['features']
        
        # Aggregate the feature grid using the desired method
        if method == 'avg':
            wsi_features = np.nanmean(feature_grid, axis=(0,1)).reshape(-1,1).T
        elif method == 'max':
            wsi_features = np.nanmax(feature_grid, axis=(0,1)).reshape(-1,1).T
        else:
            wsi_features = feature_grid.flatten()  # If you want the entire grid

        row = {
            'wsi_id': wsi_id,
            'label': group.iloc[0]['true_label'],
            'features': wsi_features
        }
        aggregated_data.append(row)

    aggregated_data_df = pd.DataFrame(aggregated_data)
    pickle.dump(aggregated_data_df, open(save_path, "wb"))
    return aggregated_data_df

# Approch 4: Calaculate attention weights are then used to compute a weighted average of the patch-level feature vectors.
# NN that outputs an attention weight for each feature vector/patch. 
class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.batch_norm = nn.BatchNorm1d(input_dim)  

        # Initializing with He initialization
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.batch_norm(x)
        x = F.relu(self.fc1(x))
        scores = self.fc2(x)
        weights = torch.softmax(scores, dim=0)
        return weights.squeeze(-1)
    
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_attention_model(train_patches_df, num_classes, num_epochs=10, lr=0.001):
    fv_dim = len(train_patches_df['features'].iloc[0])

    attention_model = AttentionModel(input_dim=fv_dim, hidden_dim=128)
    classification_head = ClassificationHead(input_dim=fv_dim, num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention_model.to(device)
    classification_head.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(attention_model.parameters()) + list(classification_head.parameters()), lr=lr)

    # Adding learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epochs):
        for wsi_id, group in train_patches_df.groupby('wsi_id'):
            features = np.vstack(group['features'].values).reshape(-1, fv_dim)
            labels = group['true_label'].values[0]

            features = torch.Tensor(features).to(device)
            labels = torch.LongTensor([labels]).to(device)

            optimizer.zero_grad()

            weights = attention_model(features)
            weighted_features = (features * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)

            output = classification_head(weighted_features)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

        scheduler.step()

    return attention_model

def aggregate_fv_with_attention(patches_df, attention_model, save_path):
    features_results = []

    for wsi_id, group in patches_df.groupby('wsi_id'):
        features = np.vstack(group['features'].values).reshape(-1, c.FEATURE_VECTOR_SIZE)
        weights = attention_model(torch.Tensor(features))
        wsi_features = (weights.unsqueeze(-1).detach().numpy() * features).sum(axis=0, keepdims=True)

        features_row = {
            'wsi_id': wsi_id,
            'label': group.iloc[0]['true_label'],
            'features': wsi_features
        }
        features_results.append(features_row)

    aggregated_data_df = pd.DataFrame(features_results)

    pickle.dump(aggregated_data_df, open(save_path, "wb"))

    return aggregated_data_df

if __name__ == '__main__':

    # Load predictions and feature vectors of indiviudal patches to create a slide-level feature vector
    method = 'avg'
    top_percent = 0.55

    train_pd = misc.load_predictions_features(c.TRAIN_PREDICTIONS, c.TRAIN_FEATURES)
    train_agg_fv = pool_top_features(train_pd, c.TRAIN_AGG_FV, method, top_percent=top_percent)

    valid_pd = misc.load_predictions_features(c.VALID_PREDICTIONS, c.VALID_FEATURES)
    valid_agg_fv = pool_top_features(valid_pd, c.VALID_AGG_FV, method, top_percent=top_percent)

    test_pd = misc.load_predictions_features(c.TEST_PREDICTIONS, c.TEST_FEATURES)
    test_agg_fv = pool_top_features(test_pd, c.TEST_AGG_FV, method, min_patches=1, top_percent=top_percent) 

    plot_features(test_pd, c.FILES_PATH, num_patches=100)

    print(f'Completed aggregating feature vectors using {method} method using the top {int(top_percent*100)}% confident patches per slide.')