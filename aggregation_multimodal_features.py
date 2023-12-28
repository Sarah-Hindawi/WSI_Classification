import misc
import config as c
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def concatenate_feature_vectors(feature_vectors, clinical_df, save_path, labels_dict, integrate_cols=[], age_factor=5, location_factor=2):
    aggregated_data = []
    
    for ind in feature_vectors.index:
        wsi_id = feature_vectors.at[ind, 'wsi_id']
        features = feature_vectors.at[ind, 'features']

        pathology_num = misc.get_pathology_num_from_labels(wsi_id, labels_dict, match_labels=True)

        row = clinical_df[clinical_df['Pathology Number'] == pathology_num]

        if not row.empty:

            agedx = row['agedx_scaled'].values[0]

            location_cols = [col for col in clinical_df.columns if 'location_ind_' in col]
            location = row[location_cols].values.flatten()

            label = row['Label'].values[0]
            
            repeated_agedx = np.tile(agedx, (1, age_factor)).flatten() 
            repeated_location = np.tile(location, location_factor)
            features_flattened = features.flatten() 

            if 'agedx' in integrate_cols:
                # Concatenate with feature vectors
                features_flattened = np.hstack([features_flattened, repeated_agedx])
            
            if 'location' in integrate_cols:
                features_flattened = np.hstack([features_flattened, repeated_location])

            aggregated_data.append({
                'wsi_id': wsi_id,
                'label': label,
                'features': features_flattened
            })

        else:
            print(f'Missing data for slide {wsi_id}. Skipping...')

    aggregated_data_df = pd.DataFrame(aggregated_data)
    pickle.dump(aggregated_data_df, open(save_path, "wb"))

    return aggregated_data_df

class AttentionModel(nn.Module):
    """
        NN that outputs an attention weight for each patch. 
        The weights are then used to compute a weighted average of the patch-level features.
        Learns to assign importance to patches, which is not directly supervised by specific labels.
    """
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
        features = np.vstack(group['features'].values).reshape(-1,512)
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
            output = attention_model(weighted_features)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

    return attention_model

def aggregate_feature_vectors(method = 'concatenate', integrate_cols=[], drop_cols=[], age_factor=6, location_factor=4):

    labels_df = pd.read_excel(c.LABELS_PATH)
    labels_dict = dict(zip(labels_df['Pathology Number'].str.strip(), labels_df['Label'].str.strip()))
  
    clinical_df = pd.read_excel(c.LABELS_PATH)
    clinical_df = clinical_df[clinical_df['Label'].notna()]
    clinical_df = clinical_df[clinical_df['Label'].isin(c.CLASSES)]


    train_fv = pd.read_pickle(c.TRAIN_AGG_FV)
    valid_fv = pd.read_pickle(c.VALID_AGG_FV)
    test_fv = pd.read_pickle(c.TEST_AGG_FV)

    slide_ids = [slide.split()[0] for slide in train_fv['wsi_id'].values]

    # Drop columns where agedx or location columns are missing
    for col in drop_cols:
        print('Dropping rows with missing:', col)
        clinical_df = clinical_df[clinical_df[col].notna()]

    if 'agedx' in integrate_cols:
        print('Integrating column agedx with factor:', age_factor)

    if 'location' in integrate_cols:
        print('Integrating column location with factor:', location_factor)

        # One-hot encode the location column
        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit(clinical_df[clinical_df['Pathology Number'].isin(train_fv['wsi_id'].values)][['location']])
        encoded_locations = encoder.transform(clinical_df[['location']])
        encoded_df = pd.DataFrame(encoded_locations, index=clinical_df.index, columns=[f"location_ind_{i}" for i in range(encoded_locations.shape[1])])
        clinical_df = pd.concat([clinical_df, encoded_df], axis=1)

    # Scale the age column
    scaler = StandardScaler()
    scaler.fit(clinical_df[clinical_df['Pathology Number'].isin(train_fv['wsi_id'].values)][['agedx']])
    clinical_df['agedx_scaled'] = scaler.transform(clinical_df[['agedx']])

    train_fv = concatenate_feature_vectors(train_fv, clinical_df, c.TRAIN_AGG_MULTI_FV, labels_dict, integrate_cols = integrate_cols, age_factor=age_factor, location_factor=location_factor)	
    valid_fv = concatenate_feature_vectors(valid_fv, clinical_df, c.VALID_AGG_MULTI_FV, integrate_cols = integrate_cols, age_factor=age_factor, location_factor=location_factor)
    test_fv = concatenate_feature_vectors(test_fv, clinical_df, c.TEST_AGG_MULTI_FV, labels_dict, integrate_cols = integrate_cols, age_factor=age_factor, location_factor=location_factor)

    return train_fv, valid_fv, test_fv

if __name__ == '__main__':
    aggregate_feature_vectors(integrate_cols=['agedx', 'location'], age_factor=5, location_factor=4)
