import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import misc
import config as c


def concatenate_feature_vectors(feature_vectors, clinical_df, save_path, integerate=True):
    aggregated_data = []

    min_agedx = clinical_df['agedx'].min()
    max_agedx = clinical_df['agedx'].max()

    location_cols = pd.get_dummies(clinical_df['location'])
    
    for ind in feature_vectors.index:
        wsi_id = feature_vectors.at[ind, 'wsi_id']
        features = feature_vectors.at[ind, 'features']

        row = clinical_df[clinical_df['Pathology Number'] == wsi_id]

        if not row.empty:
            agedx = row['agedx'].values[0]
            agedx = (agedx - min_agedx) / (max_agedx - min_agedx) # Normalize

            label = row['Label'].values[0]

            one_hot_encoded = location_cols.loc[row.index].values.flatten()
            
            repeat_factor = 10
            repeated_agedx = np.tile(agedx, (1, repeat_factor)).flatten() 
            repeated_location = np.tile(one_hot_encoded, repeat_factor)
            features_flattened = features.flatten() 

            if integerate:
                # Concatenate with feature vectors
                final_feature_vector = np.hstack([repeated_agedx, repeated_location])
            else:
                final_feature_vector = features_flattened

            aggregated_data.append({
                'wsi_id': wsi_id,
                'label': label,
                'features': final_feature_vector
            })

    aggregated_data_df = pd.DataFrame(aggregated_data)
    pickle.dump(aggregated_data_df, open(save_path, "wb"))

# NN that outputs an attention weight for each patch. 
# The weights are then used to compute a weighted average of the patch-level features.
# Learns to assign importance to patches, which is not directly supervised by specific labels.
class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # output a score for each patch
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        scores = self.fc2(x)  # output raw scores
        weights = torch.softmax(scores, dim=0)  # convert scores to weights
        return weights.squeeze(-1)

def average_weighted_fv(patches_df, save_path, attention_model):

    features_results = []

    for wsi_id, group in patches_df.groupby('wsi_id'):
        features = np.vstack(group['features'].values).reshape(-1,512) # Reshape features
        weights = attention_model(torch.Tensor(features))  # Compute attention weights
        wsi_features = np.average(features, weights=weights.detach().numpy(), axis=0).reshape(-1,1).T  # Compute weighted average of features

        features_row = {
            'wsi_id': wsi_id,
            'label': group.iloc[0]['True Label'],
            'features': wsi_features
        }
        features_results.append(features_row)

    features_results_df = pd.DataFrame(features_results)
    pickle.dump(features_results_df, open(save_path, "wb"))  # Save the aggregated features

    return features_results_df

def train_attention_model(train_patches_df, num_epochs=10, lr=0.001):
    fv_dim = train_patches_df['features'].iloc[0].shape[1]

    attention_model = AttentionModel(input_dim=fv_dim, hidden_dim=128) 

    device = misc.get_device()
    attention_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(attention_model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        for wsi_id, group in train_patches_df.groupby('wsi_id'):
            features = np.vstack(group['features'].values).reshape(-1,fv_dim)
            labels = group['true_label'].values[0]
            # Convert to torch tensors and move to appropriate device
            features = torch.Tensor(features).to(device)
            labels = torch.LongTensor([labels]).to(device)
            
            optimizer.zero_grad()

            weights = attention_model(features)
            weighted_features = (features * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)  # weighted sum of features
            output = attention_model(weighted_features)  # You'll need a separate classification model here
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

    return attention_model

def main():
    clinical_df = pd.read_excel(c.LABELS_PATH)
    clinical_df = clinical_df[clinical_df['Label'].notna()]

    # Check if the 'subtype' column matches any of the required classes
    clinical_df = clinical_df[clinical_df['Label'].isin(c.CLASSES)]

    train_fv = pd.read_pickle(c.TRAIN_AGG_FV)
    valid_fv = pd.read_pickle(c.VALID_AGG_FV)
    test_fv = pd.read_pickle(c.TEST_AGG_FV)

    integerate = False
    concatenate_feature_vectors(train_fv, clinical_df, c.TRAIN_AGG_MULTI_FV, integerate = integerate)
    concatenate_feature_vectors(valid_fv, clinical_df, c.VALID_AGG_MULTI_FV, integerate = integerate)
    concatenate_feature_vectors(test_fv, clinical_df, c.TEST_AGG_MULTI_FV, integerate = integerate)

if __name__ == '__main__':
    main()
