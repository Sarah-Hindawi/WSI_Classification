import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import misc
import config as c

# Approach 1: Average all the feature vectors of the patches in a slide. 
# Provides a general representation of the slide, but might overlook key/tumor regions
def pool_features(patches_df, save_path, method='avg'):

    aggregated_data = []

    for wsi_id, group in patches_df.groupby('wsi_id'):
        if method == 'avg':
            wsi_features = np.mean(np.vstack(group['features'].values).reshape(-1,c.FEATURE_VECTOR_SIZE), axis=0).reshape(-1,1).T # Compute mean of features
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

# NN that outputs an attention weight for each feature vector/patch. 
# The weights are then used to compute a weighted average of the patch-level feature vectors.
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

# Rest of the code remains unchanged

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

# if __name__ == '__main__':

#     # Load predictions and feature vectors of indiviudal patches to create a slide-level feature vector
#     method = 'avg'
#     train_pd = misc.load_predictions_features(c.TRAIN_PREDICTIONS, c.TRAIN_FEATURES)
#     train_agg_fv = pool_features(train_pd, c.TRAIN_AGG_FV, method)

#     valid_pd = misc.load_predictions_features(c.VALID_PREDICTIONS, c.VALID_FEATURES)
#     valid_agg_fv = pool_features(valid_pd, c.VALID_AGG_FV, method)

#     test_pd = misc.load_predictions_features(c.TEST_PREDICTIONS, c.TEST_FEATURES)
#     test_agg_fv = pool_features(test_pd, c.TEST_AGG_FV, method) 


if __name__ == '__main__':


    train_pd = misc.load_predictions_features(c.TRAIN_PREDICTIONS, c.TRAIN_FEATURES)
    attention_model = train_attention_model(train_pd, len(c.CLASSES))

    train_agg_fv = aggregate_fv_with_attention(train_pd, attention_model, c.TRAIN_AGG_FV)

    valid_pd = misc.load_predictions_features(c.VALID_PREDICTIONS, c.VALID_FEATURES)
    valid_agg_fv = aggregate_fv_with_attention(valid_pd, attention_model, c.VALID_AGG_FV)

    test_pd = misc.load_predictions_features(c.TEST_PREDICTIONS, c.TEST_FEATURES)
    test_agg_fv = aggregate_fv_with_attention(test_pd, attention_model, c.TEST_AGG_FV)
