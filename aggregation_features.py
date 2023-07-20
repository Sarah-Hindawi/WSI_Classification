import os
import torch
import numpy as np
import pickle
import pandas as pd
from sklearn.svm import SVC
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# NN that outputs an attention weight for each patch. 
# The weights are then used to compute a weighted average of the patch-level features.
# Learns to assign importance to patches, which is not directly supervised by specific labels.
class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(AttentionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)  # output a score for each class
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # output raw scores (logits)

def create_svm(train_features, valid_features, test_features):
    train_feature_vectors = np.concatenate(train_features['Features'].tolist()).reshape(-1, 512)

    # Train SVM model on feature vectors extracted from the CNN previously
    clf = SVC(probability=True)
    clf.fit(train_feature_vectors, train_features['Label'])

    # Evaluate on test feature vectors and output predictions
    test_feature_vectors = np.concatenate(test_features['Features'].tolist()).reshape(-1, 512)
    prob_preds = clf.predict_proba(test_feature_vectors)
    class_preds = clf.predict(test_feature_vectors)

    print("Predicted probabilities: ")
    print(prob_preds)
    print("Predicted class labels: ")
    print(class_preds)
    print("True labels: ")
    print(test_features['Label'])

    accuracy = accuracy_score(test_features['Label'], class_preds)
    print("Accuracy on the test set: ", accuracy)

    return prob_preds, class_preds


def load_data(predictions_path, features_path):
    preds = pd.read_excel(predictions_path)
    features = pd.DataFrame(pd.read_pickle(features_path))

    patches_df = preds.merge(features, on=["WSI_id", "Patch_index"]) # merge tables horizantally
    return patches_df

# Approach 1: Average all the feature vectors of the patches in a slide. 
# Provides a general representation of the slide, but might overlook key/tumor regions
def average_fv(patches_df, save_path):

    features_results = []

    for wsi_id, group in patches_df.groupby('WSI_id'):
        wsi_features = np.mean(np.vstack(group['Features'].values).reshape(-1,512), axis=0).reshape(-1,1).T # Compute mean of features
        features_row = {
            'WSI_id': wsi_id,
            'Label': group.iloc[0]['True Label'],
            'Features': wsi_features
        }
        features_results.append(features_row)

    features_results_df = pd.DataFrame(features_results)
    pickle.dump(features_results_df, open(train_save_path, "wb"))  # Save the aggregated features

    return features_results_df

def average_weighted_fv(patches_df, save_path, attention_model):

    features_results = []

    for wsi_id, group in patches_df.groupby('WSI_id'):
        features = np.vstack(group['Features'].values).reshape(-1,512) # Reshape features
        weights = attention_model(torch.Tensor(features))  # Compute attention weights
        wsi_features = np.average(features, weights=weights.detach().numpy(), axis=0).reshape(-1,1).T  # Compute weighted average of features

        features_row = {
            'WSI_id': wsi_id,
            'Label': group.iloc[0]['True Label'],
            'Features': wsi_features
        }
        features_results.append(features_row)

    features_results_df = pd.DataFrame(features_results)
    pickle.dump(features_results_df, open(train_save_path, "wb"))  # Save the aggregated features

    return features_results_df

def train_attention_model(train_patches_df, num_epochs=10, lr=0.001):
    num_classes = len(train_patches_df['True Label'].unique())
    fv_dim = train_patches_df['Features'].iloc[0].shape[1]

    attention_model = AttentionModel(input_dim=fv_dim, hidden_dim=128, num_classes=num_classes) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(attention_model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        for wsi_id, group in train_patches_df.groupby('WSI_id'):
            features = np.vstack(group['Features'].values).reshape(-1,fv_dim)
            labels = group['True Label'].values[0] 
            # Convert to torch tensors and move to appropriate device
            features = torch.Tensor(features).to(device)
            labels = torch.LongTensor([labels]).to(device)
            
            optimizer.zero_grad()
            output = attention_model(features)
            output_fv = output.mean(dim=0, keepdim=True)  # aggregate the outputs by taking the mean
            loss = criterion(output_fv, labels)

            loss.backward()
            optimizer.step()

    return attention_model

if __name__ == '__main__':

    files_path = 'hpf files'

    train_predictions_path = os.path.join(files_path, 'train_predictions.xlsx')
    train_features_path = os.path.join(files_path, 'train_features.pkl')
    train_save_path = os.path.join(files_path, 'train_feature_vector')
    train_pd = load_data(train_predictions_path, train_features_path)
    # train_features = average_weighted_fv(train_pd, train_save_path)

    train_attention_model(train_pd)

    print("\nValidation set:")
    valid_predictions_path = os.path.join(files_path, 'validation_predictions.xlsx')
    valid_features_path = os.path.join(files_path, 'validation_features.pkl')
    valid_save_path = os.path.join(files_path, 'valid_WSI_classification.xlsx')
    valid_pd = load_data(valid_predictions_path, valid_features_path)
    valid_features = average_weighted_fv(valid_pd, valid_save_path, train_attention_model)

    print("\nTest set:")
    test_predictions_path = os.path.join(files_path, 'test_predictions.xlsx')
    test_features_path = os.path.join(files_path, 'test_features.pkl')
    test_save_path = os.path.join(files_path, 'test_WSI_classification.xlsx')
    test_pd = load_data(test_predictions_path, test_features_path)
    test_features = average_weighted_fv(test_pd, test_save_path, train_attention_model)

    # create_svm(train_features, valid_features, test_features)


