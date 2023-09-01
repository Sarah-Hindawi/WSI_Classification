import pickle
import numpy as np
import pandas as pd

import misc
import config as c

# Approach 1: Average all the feature vectors of the patches in a slide. 
# Provides a general representation of the slide, but might overlook key/tumor regions
def pool_features(patches_df, save_path, method='avg'):

    aggregated_features = []

    for wsi_id, group in patches_df.groupby('wsi_id'):
        if method == 'avg':
            wsi_features = np.mean(np.vstack(group['features'].values).reshape(-1,512), axis=0).reshape(-1,1).T # Compute mean of features
        elif method == 'max':
            wsi_features = np.max(np.vstack(group['features'].values).reshape(-1, 512), axis=0).reshape(-1,1).T

        features_row = {
            'wsi_id': wsi_id,
            'label': group.iloc[0]['true_label'],
            'features': wsi_features
        }
        aggregated_features.append(features_row)

    aggregated_features_df = pd.DataFrame(aggregated_features)
    pickle.dump(aggregated_features_df, open(save_path, "wb"))  # Save the aggregated features

    return aggregated_features_df

if __name__ == '__main__':

    # Load predictions and feature vectors of indiviudal patches to create a slide-level feature vector
    method = 'avg'
    train_pd = misc.load_predictions_features(c.TRAIN_PREDICTIONS, c.TRAIN_FEATURES)
    train_agg_fv = pool_features(train_pd, c.TRAIN_AGG_FV, method)

    valid_pd = misc.load_predictions_features(c.VALID_PREDICTIONS, c.VALID_FEATURES)
    valid_agg_fv = pool_features(valid_pd, c.VALID_AGG_FV, method)

    test_pd = misc.load_predictions_features(c.TEST_PREDICTIONS, c.TEST_FEATURES)
    test_agg_fv = pool_features(test_pd, c.TEST_AGG_FV, method) 