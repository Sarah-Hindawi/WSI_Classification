import pickle
import random
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import config as c

from umap import UMAP
from scipy.stats import mode
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, accuracy_score, roc_auc_score, log_loss
from sklearn.metrics import precision_recall_fscore_support, silhouette_score, confusion_matrix, classification_report
from aggregation_multimodal_features import aggregate_feature_vectors

random.seed(42)
np.random.seed(42)

def svm_classifier(x_train, y_train, x_valid, y_valid, x_test, y_test, wsi_ids_test=None, label_encoder=None):

    C_values = [0.0001, 0.001, 0.01, 0.1, 0.2]
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    parameters = {'C': C_values, 'kernel': kernels}

    clf = SVC(probability=True, class_weight='balanced', random_state=42)
    clf = GridSearchCV(clf, parameters)
    clf.fit(x_train, y_train)

    best_params = clf.best_params_

    valid_preds = clf.predict(x_valid)   
    valid_prob_preds = clf.predict_proba(x_valid)   
    current_log_loss = log_loss(y_valid, valid_prob_preds)
    
    print(f"Validation Accuracy {accuracy_score(y_valid, valid_preds)}. Log loss with C={best_params['C']}, kernel={best_params['kernel']} on validation set: {current_log_loss:.3f}")

    test_preds = clf.predict(x_test)    
    test_prob_preds = clf.predict_proba(x_test)

    display_results(y_test, test_preds, test_prob_preds, wsi_ids_test, label_encoder)

    # Save best model for future inference
    pickle.dump(clf, open(c.SVM_MODEL, "wb"))

    return clf

def random_forest_classifier(x_train, y_train, x_valid, y_valid, x_test, y_test, wsi_ids_test=None, label_encoder=None):

    n_estimators = [10, 15, 20, 30, 40, 50, 100, 200, 300]
    max_depth = [None, 20]
    max_features = ['sqrt']
    parameters = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features}

    parameters = {'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 80, 90,  None],
    'max_features': [None, 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', 'balanced_subsample'],
    'n_estimators': n_estimators}

    clf = RandomForestClassifier(class_weight='balanced', random_state=42)
    clf = GridSearchCV(clf, parameters)
    clf.fit(x_train, y_train)

    best_params = clf.best_params_

    valid_preds = clf.predict(x_valid)   
    valid_prob_preds = clf.predict_proba(x_valid)
    current_log_loss = log_loss(y_valid, valid_prob_preds)
    
    print(f"Validation Accuracy {accuracy_score(y_valid, valid_preds)}. Best number of estimators: {best_params['n_estimators']}, best max_depth: {best_params['max_depth']} with log loss: {current_log_loss:.3f}")

    test_preds = clf.predict(x_test)       
    test_prob_preds = clf.predict_proba(x_test)

    display_results(y_test, test_preds, test_prob_preds, wsi_ids_test, label_encoder)

    pickle.dump(clf, open(c.RANDOM_FOREST_MODEL, "wb"))

    return clf

def xgboost_classifier(x_train, y_train, x_valid, y_valid, x_test, y_test, wsi_ids_test=None, label_encoder=None):

    learning_rate = np.arange(0.0001, 0.035, 0.001)
    max_depth = [1, 2, 3, 4, 5]
    booster = ['dart']
    rate_drop = [0, 0.3, 0.5]

    parameters = {
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'rate_drop': rate_drop
    }
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    clf = xgb.XGBClassifier(eval_metric="mlogloss", num_class=len(c.CLASSES), early_stopping_rounds=10, random_state=42)
    clf = GridSearchCV(clf, parameters, cv=stratified_kfold, n_jobs=-1)
    clf.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)

    best_params = clf.best_params_

    valid_preds = clf.predict(x_valid)   
    valid_prob_preds = clf.predict_proba(x_valid)
    valid_loss = log_loss(y_valid, valid_prob_preds)
    
    print(f"Validation Accuracy {accuracy_score(y_valid, valid_preds)}. Best parameters: eta={best_params['learning_rate']:.6f}, max_depth={best_params['max_depth']} with validation log loss: {valid_loss:.3f}")

    test_preds = clf.predict(x_test)
    test_prob_preds = clf.predict_proba(x_test)

    display_results(y_test, test_preds, test_prob_preds, wsi_ids_test, label_encoder)

    pickle.dump(clf, open(c.XGBOOST_MODEL, "wb"))

    return clf

def load_predict(model_path, x_test, y_test, wsi_ids_test=None, verbose=False):
    clf = pickle.load(open(model_path, "rb"))

    test_preds = clf.predict(x_test)
    test_prob_preds = clf.predict_proba(x_test)

    display_results(y_test, test_preds, test_prob_preds, wsi_ids_test, verbose)

    return clf

def cluster(x_train, y_train, x_valid, y_valid, x_test, y_test):
    matplotlib.use('Agg')

    pca_combined_data, pca_x_test = apply_pca(x_train, y_train, x_valid, y_valid, x_test, y_test)
    
    # Train KMeans on the PCA transformed combined data
    kmeans = KMeans(n_clusters=len(c.CLASSES), init='k-means++', n_init=10, random_state=42)
    kmeans.fit(pca_combined_data)

    cluster_labels = kmeans.predict(pca_x_test)

    plt.scatter(pca_x_test[:, 0], pca_x_test[:, 1], c=cluster_labels, cmap='rainbow')
    plt.title('Clusters using K-means on Test Data')
    plt.savefig(c.KMEANS_CLUSTER_PATH)
    plt.clf()
    plt.close()

    # Check if we get all samples in one cluster
    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) == 1:
        print(f"All samples assigned to cluster {unique_labels[0]}")

    return cluster_labels, y_test

def cluster_dbscan(x_train, y_train, x_valid, y_valid, x_test, y_test):
    combined_data = np.vstack((x_train, x_valid))
    

    pca = PCA(n_components=0.95) 
    pca_combined_data = pca.fit_transform(combined_data)
    pca_x_test = pca.transform(x_test)
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    cluster_labels = dbscan.fit_predict(pca_x_test)
    
    # Check if we still get all samples in one cluster
    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) == 1:
        print(f"All samples assigned to cluster {unique_labels[0]}")
    else:
        # Visualizing clusters
        plt.scatter(pca_x_test[:, 0], pca_x_test[:, 1], c=cluster_labels, cmap='rainbow')
        plt.title('Clusters using DBSCAN')
        plt.savefig(c.DBSCAN_CLUSTER_PATH)

        # Evaluating clusters using Silhouette score and Adjusted Rand Index
        silhouette = silhouette_score(pca_x_test, cluster_labels)
        ari = adjusted_rand_score(y_test, cluster_labels)

        print(f"Silhouette Score: {silhouette}")
        print(f"Adjusted Rand Index: {ari}")


def apply_pca(x_train, y_train, x_valid, y_valid, x_test, y_test):
    # Combining training and validation sets to utilize the validation set in clustering
    x_train_combined = np.vstack((x_train, x_valid))
    y_train_combined = np.hstack((y_train, y_valid))
    
    scaler = StandardScaler()
    x_train_combined = scaler.fit_transform(x_train_combined)
    x_test = scaler.transform(x_test)
    
    pca = PCA(n_components=0.95, random_state=42)  # Preserve 95% of variance
    pca_combined_data = pca.fit_transform(x_train_combined)
    pca_x_test = pca.transform(x_test)

    # Visualizing PCA-transformed samples with true labels for training and test data
    plt.scatter(pca_combined_data[:, 0], pca_combined_data[:, 1], c=y_train_combined, cmap='rainbow', s=5)
    plt.title('PCA Transformed Training Samples (True Labels)')
    plt.savefig(c.PCA_TRAIN_LABEL_PATH)
    plt.clf()
    plt.close()

    plt.scatter(pca_x_test[:, 0], pca_x_test[:, 1], c=y_test, cmap='rainbow', s=5)
    plt.title('PCA Transformed Test Samples (True Labels)')
    plt.savefig(c.PCA_TEST_LABEL_PATH)
    plt.clf()
    plt.close() 

    return pca_combined_data, pca_x_test

def apply_umap(x_train, y_train, x_valid, y_valid, x_test, y_test):
    # Combining training and validation sets to utilize the validation set in clustering
    x_train_combined = np.vstack((x_train, x_valid))
    y_train_combined = np.hstack((y_train, y_valid))
    
    # Applying UMAP
    umap = UMAP(n_components=2, random_state=42)
    umap_combined_data = umap.fit_transform(x_train_combined)
    umap_x_test = umap.transform(x_test)

    # Visualizing UMAP-transformed samples with true labels for training and test data
    plt.scatter(umap_combined_data[:, 0], umap_combined_data[:, 1], c=y_train_combined, cmap='rainbow', s=5)
    plt.title('UMAP Transformed Training Samples (True Labels)')
    plt.savefig(c.UMAP_TRAIN_LABEL_PATH)
    plt.clf()
    plt.close()

    plt.scatter(umap_x_test[:, 0], umap_x_test[:, 1], c=y_test, cmap='rainbow', s=5)
    plt.title('UMAP Transformed Test Samples (True Labels)')
    plt.savefig(c.UMAP_TEST_LABEL_PATH)
    plt.clf()
    plt.close()

    return umap_combined_data, umap_x_test

def display_results(y_test, preds, test_prob_preds, wsi_ids_test=None, label_encoder=None, verbose=False):
    accuracy = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, test_prob_preds, multi_class='ovr')
        
    print("Accuracy on the test set: ", accuracy, "\nROC AUC score: ", roc_auc)
    print("Pred =", list(preds))
    print("True =", list(y_test))

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, preds, average='weighted')

    print(f"Precision (weighted average): {precision:.2f}")
    print(f"Recall (weighted average): {recall:.2f}")
    print(f"F1-Score (weighted average): {f1_score:.2f}")

		
    # Count the number of misclassified and correctly classified samples for each class
    correct = np.zeros(len(c.CLASSES)).astype(int)
    total = np.zeros(len(c.CLASSES)).astype(int)
    for i in range(len(y_test)):
        total[y_test[i]] += 1
        if y_test[i] == preds[i]:
            correct[y_test[i]] += 1
    for i in range(len(c.CLASSES)):
        label = next(value for key, value in label_encoder.items() if key == i)
        print(f"\nPercentage of {label} samples correctly classified: {100*(correct[i]/total[i]):.2f}%")
        print(f"Correctly classified samples = {correct[i]}\nMisclassified samples = {total[i]-correct[i]}")

    if verbose: 
        print("Correctly classified samples:") if (y_test == preds).any() else print("No correctly classified samples.")
        for i in range(len(y_test)):
            if y_test[i] == preds[i]:
                print(f'{wsi_ids_test[i]}: Label: {label_encoder[y_test[i]]}, Probs: {test_prob_preds[i]}')


        print("\nMisclassified samples:") if (y_test != preds).any() else print("No misclassified samples.")
        for i in range(len(y_test)):
            if y_test[i] != preds[i]:
                print(f'{wsi_ids_test[i]}: Predicted: {label_encoder[preds[i]]}, True: {label_encoder[y_test[i]]}, Probs: {test_prob_preds[i]}')

    print(classification_report(y_test, preds))

    save_confusion_matrix(y_test, preds)

def save_confusion_matrix(y_test, preds):
    conf_matrix = confusion_matrix(y_test, preds)
    labels = ['BRAF SNV', 'BRAF fusion', 'FGFR Altered']

    # Create a mask for correctly/incorrectly predicted cells
    mask = np.zeros_like(conf_matrix, dtype=bool)
    mask[np.diag_indices_from(mask)] = True
    
    # Custom colormaps
    green_cmap = ListedColormap(['white', 'lightgreen', 'mediumseagreen', 'green'])
    red_cmap = ListedColormap(['white', 'lightcoral', 'salmon', 'red'])
    
    # Set the normalization
    green_norm = [0, 1, 10, 15, 20]
    red_norm = [0, 1, 2, 10, 20]
    
    # Use the mask in the heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', mask=~mask, cmap=green_cmap, norm=matplotlib.colors.BoundaryNorm(green_norm, green_cmap.N), xticklabels=labels, yticklabels=labels)
    sns.heatmap(conf_matrix, annot=True, fmt='d', mask=mask, cmap=red_cmap, norm=matplotlib.colors.BoundaryNorm(red_norm, red_cmap.N), xticklabels=labels, yticklabels=labels)

    # Set cells with 0 cases to white
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        if conf_matrix[i, j] == 0:
            plt.gca().add_patch(Rectangle((j, i), 1, 1, fill=True, color='white', lw=0))

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Genetic Mutation Prediction')

    plt.savefig(c.CONFUSION_MATRIX_PATH)
    plt.close()
    
def reassign_labels(cluster_labels, true_labels):
    """
    Reassign cluster labels based on mode of true class labels in each cluster.

    Parameters:
    - cluster_labels: List of predicted cluster labels
    - true_labels: List of true class labels

    Returns:
    - new_cluster_labels: List of reassigned cluster labels
    """
    unique_clusters = np.unique(cluster_labels)
    
    # Placeholder for new cluster labels
    new_cluster_labels = np.zeros(cluster_labels.shape, dtype=np.int64)
    
    label_map = {}

    for cluster in unique_clusters:
        # Mask to select items of this cluster
        mask = cluster_labels == cluster
        
        # Determine the mode of true labels in this cluster
        true_mode = mode(true_labels[mask])[0]
        
        # Assign this mode as the new label for this cluster
        label_map[cluster] = true_mode
        
        new_cluster_labels[mask] = true_mode
        
    print(f"Cluster label mapping: {label_map}")
    return new_cluster_labels

if __name__ == '__main__':

    integrate_cols = []
    drop_cols = []

    age_factors = [2,3,4,5,6,7,8,9]
    location_factors = [2,3,4,5,6,7,8]

    if not integrate_cols:
        age_factors = location_factors = [1] # Do not loop over factor values

    attempted_combinations = []

    for age in age_factors:
        for location in location_factors:

            comb = (age, location)
            if comb in attempted_combinations:
                continue
            attempted_combinations.append(comb)

            aggregate_feature_vectors(integrate_cols=integrate_cols, drop_cols=drop_cols, age_factor = age, location_factor = location)

            train_features = pd.DataFrame(pd.read_pickle(c.TRAIN_AGG_MULTI_FV))
            valid_features = pd.DataFrame(pd.read_pickle(c.VALID_AGG_MULTI_FV))
            test_features = pd.DataFrame(pd.read_pickle(c.TEST_AGG_MULTI_FV))

            # Flatten and concatenate all feature vectors
            sample_length = len(train_features['features'].iloc[0])

            x_train = np.concatenate(train_features['features'].tolist()).reshape(-1, sample_length)
            x_valid = np.concatenate(valid_features['features'].tolist()).reshape(-1, sample_length)
            x_test = np.concatenate(test_features['features'].tolist()).reshape(-1, sample_length)

            imputer = IterativeImputer()
            x_train = imputer.fit_transform(x_train)
            x_valid = imputer.transform(x_valid)
            x_test = imputer.transform(x_test)

            scaler = StandardScaler() # Feature Scaling
            x_train = scaler.fit_transform(x_train)
            x_valid = scaler.transform(x_valid)
            x_test = scaler.transform(x_test)

            target_name = 'label'
            y_train = train_features[target_name]
            y_valid = valid_features[target_name]
            y_test = test_features[target_name]

            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_valid = label_encoder.transform(y_valid)
            y_test = label_encoder.transform(y_test)

            label_encoder_map = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))

            print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

            wsi_ids_test = test_features['wsi_id'].tolist() # To display the WSIs that were misclassified

            x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, stratify=y_train, random_state=42)

            model = ['random forest', 'cluster']
            
            if 'svm' in model:
                print('Classification using SVM:')
                svm_classifier(x_train, y_train, x_valid, y_valid, x_test, y_test, wsi_ids_test, label_encoder_map)

            if 'random forest' in model:
                print('\n\nClassification using Random Forest:')
                random_forest_classifier(x_train, y_train, x_valid, y_valid, x_test, y_test, wsi_ids_test, label_encoder_map)

            if 'xgboost' in model:
                print('\n\nClassification using XGBoost:')
                xgboost_classifier(x_train, y_train, x_valid, y_valid, x_test, y_test, wsi_ids_test, label_encoder_map)

            if 'cluster' in model:
                cluster_labels = cluster(x_train, y_train, x_valid, y_valid, x_test, y_test)
                print(cluster_labels)

            # load_predict(c.RANDOM_FOREST_MODEL, x_test, y_test, wsi_ids_test, verbose=False)
