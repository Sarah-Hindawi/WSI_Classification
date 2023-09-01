import os
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

import misc
import config as c

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

if __name__ == '__main__':

    files_path = c.FILES_PATH

    train_predictions_path = os.path.join(files_path, c.TRAIN_PREDICTIONS)
    train_features_path = os.path.join(files_path, c.TRAIN_FEATURES)
    train_save_path = os.path.join(files_path, c.TRAIN_AGG_FV)
    train_pd = misc.load_predictions_features(train_predictions_path, train_features_path)
    # train_features = average_weighted_fv(train_pd, train_save_path)

    attention_model = train_attention_model(train_pd)

    print("\nValidation set:")
    valid_predictions_path = os.path.join(files_path,  c.VALID_PREDICTIONS)
    valid_features_path = os.path.join(files_path, c.VALID_FEATURES)
    valid_save_path = os.path.join(files_path, c.VALID_WSI_CLASSIFICATION)
    valid_pd = misc.load_predictions_features(valid_predictions_path, valid_features_path)
    valid_features = average_weighted_fv(valid_pd, valid_save_path, attention_model)

    print("\nTest set:")
    test_predictions_path = os.path.join(files_path, c.TRAIN_PREDICTIONS)
    test_features_path = os.path.join(files_path, c.TRAIN_FEATURES)
    test_save_path = os.path.join(files_path, c.TEST_WSI_CLASSIFICATION)
    test_pd = misc.load_predictions_features(test_predictions_path, test_features_path)
    test_features = average_weighted_fv(test_pd, test_save_path, attention_model)

    # create_svm(train_features, valid_features, test_features)