import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, accuracy_score, roc_auc_score, log_loss, silhouette_score

import config as c

def svm_classifier(x_train, y_train, x_valid, y_valid, x_test, y_test, wsi_ids_test):

    C_values = [0.1, 1, 10]
    kernels = ['linear', 'rbf']

    best_log_loss = float('inf')
    best_params = None
    best_clf = None
    
    # Search for best hyperparameters using validation set
    for C in C_values:
        for kernel in kernels:
            clf = SVC(C=C, kernel=kernel, probability=True)
            clf.fit(x_train, y_train)

            valid_prob_preds = clf.predict_proba(x_valid)
            
            current_log_loss = log_loss(y_valid, valid_prob_preds)
            print(f"Log loss with C={C}, kernel={kernel} on validation set: {current_log_loss:.3f}")

            if current_log_loss < best_log_loss:
                best_log_loss = current_log_loss
                best_params = (C, kernel)
                best_clf = clf

    print(f"Best parameters: C={best_params[0]}, kernel={best_params[1]} with log loss: {best_log_loss}")

    # Using the best model to predict on the test set
    test_preds = best_clf.predict(x_test)    
    test_prob_preds = best_clf.predict_proba(x_test)

    display_results(y_test, test_preds, test_prob_preds)

    return best_clf

def random_forest_classifier(x_train, y_train, x_valid, y_valid, x_test, y_test, wsi_ids_test):
    n_estimators_range = range(10, 101, 10)
    
    best_log_loss = float('inf')
    best_n_estimators = None
    best_clf = None
    
    # Search for best n_estimators using validation set
    for n_estimators in n_estimators_range:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(x_train, y_train)

        valid_prob_preds = clf.predict_proba(x_valid)
        
        current_log_loss = log_loss(y_valid, valid_prob_preds)
        print(f"Log loss with {n_estimators} estimators on validation set: {current_log_loss:.3f}")

        if current_log_loss < best_log_loss:
            best_log_loss = current_log_loss
            best_n_estimators = n_estimators
            best_clf = clf

    print(f"Best number of estimators: {best_n_estimators} with log loss: {best_log_loss:.3f}")

    test_preds = best_clf.predict(x_test)    
    test_prob_preds = best_clf.predict_proba(x_test)
    
    test_preds = best_clf.predict(x_test)    
    test_prob_preds = best_clf.predict_proba(x_test)

    display_results(y_test, test_preds, test_prob_preds)

    return best_clf

def xgboost_classifier(x_train, y_train, x_valid, y_valid, x_test, y_test):

    # Range for learning rates
    learning_rate_range = np.arange(0.01, 1, 0.1)

    best_log_loss = float('inf')
    best_lr = None
    best_xgb_model = None

    log_losses = []
    
    # Iterate over learning rates
    for lr in learning_rate_range:
        xgb_classifier = xgb.XGBClassifier(eta=lr)
        xgb_classifier.fit(x_train, y_train)

        valid_prob_preds = xgb_classifier.predict_proba(x_valid)
        
        current_log_loss = log_loss(y_valid, valid_prob_preds)
        log_losses.append(current_log_loss)

        print(f"Log loss with learning rate {lr:.2f} on validation set: {current_log_loss:.3f}")

        if current_log_loss < best_log_loss:
            best_log_loss = current_log_loss
            best_lr = lr
            best_xgb_model = xgb_classifier

    # Using the best model to predict on the test set
    test_preds = best_xgb_model.predict(x_test)
    test_accuracy = accuracy_score(y_test, test_preds)
    
    print(f"Best learning rate: {best_lr:2f} with log loss: {best_log_loss}")
    print("Accuracy on the test set: ", test_accuracy)

    # Plotting
    fig = plt.figure(figsize=(10, 7))
    plt.plot(learning_rate_range, log_losses, c='blue', label='Validation Log Loss')
    plt.xlabel('Learning rate')
    plt.xticks(learning_rate_range)
    plt.ylabel('Log Loss')
    plt.legend(prop={'size': 12}, loc=3)
    plt.title('Log Loss vs. Learning rate of XGBoost', size=14)
    plt.show()
    # plt.savefig(c.XGBOOST_PLOT_PATH)

def cluster(x_train, y_train, x_valid, y_valid, x_test, y_test):
    # Combining training and validation sets to utalize the validation set in clustering
    combined_data = np.vstack((x_train, x_valid))

    # Scale the data
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data)
    x_test_scaled = scaler.transform(x_test)

    # Apply PCA 
    pca = PCA(n_components=0.95)  # Preserve 95% of variance
    pca_combined_data = pca.fit_transform(combined_data_scaled)
    pca_x_test = pca.transform(x_test_scaled)

    # Train KMeans on the scaled combined data
    kmeans = KMeans(n_clusters=len(c.CLASSES), init='k-means++', n_init=10, random_state=42)
    kmeans.fit(pca_combined_data)

    # Predict the clusters for scaled x_test
    cluster_labels = kmeans.predict(pca_x_test)

    # Visualizing clusters
    plt.scatter(pca_x_test[:, 0], pca_x_test[:, 1], c=cluster_labels, cmap='rainbow')
    plt.title('Clusters using K-means')
    plt.savefig(c.KMEANS_CLUSTER_PATH)

    # Check if we still get all samples in one cluster
    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) == 1:
        print(f"All samples assigned to cluster {unique_labels[0]}")
    else:
        # Evaluating clusters using Silhouette score and Adjusted Rand Index
        silhouette = silhouette_score(pca_x_test, cluster_labels)
        ari = adjusted_rand_score(y_test, cluster_labels)

        print(f"Silhouette Score: {silhouette}")
        print(f"Adjusted Rand Index: {ari}")

def cluster_dbscan(x_train, y_train, x_valid, y_valid, x_test, y_test):
    # Combine the training and validation sets
    combined_data = np.vstack((x_train, x_valid))
    
    # Scale the data
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data)
    x_test_scaled = scaler.transform(x_test)
    
    # Apply PCA 
    pca = PCA(n_components=0.95)  # Preserve 95% of variance
    pca_combined_data = pca.fit_transform(combined_data_scaled)
    pca_x_test = pca.transform(x_test_scaled)
    
    # DBSCAN clustering
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


def display_results(y_test, test_preds, test_prob_preds):
    accuracy = accuracy_score(y_test, test_preds)
    roc_auc = roc_auc_score(y_test, test_prob_preds, multi_class='ovr')
        
    print("Accuracy on the test set: ", accuracy, "\nROC AUC score: ", roc_auc)
    print("Predicted probabilities on test set:\n", test_prob_preds)
    print("Predicted class labels on test set:", test_preds)
    print("True labels: ", list(y_test))
    print('WSIs that were misclassified:', [wsi_ids_test[i] for i in range(len(y_test)) if y_test[i] != test_preds[i]])


if __name__ == '__main__':

    files_path = c.FILES_PATH

    train_features = pd.DataFrame(pd.read_pickle(c.TRAIN_AGG_FV))
    valid_features = pd.DataFrame(pd.read_pickle(c.VALID_AGG_FV))
    test_features = pd.DataFrame(pd.read_pickle(c.TEST_AGG_FV))

    # Flatten and concatenate all tile-level feature vectors
    x_train = np.concatenate(train_features['features'].tolist()).reshape(-1, 512)
    x_valid = np.concatenate(valid_features['features'].tolist()).reshape(-1, 512)
    x_test = np.concatenate(test_features['features'].tolist()).reshape(-1, 512)

    y_train = train_features['label']
    y_valid = valid_features['label']
    y_test = test_features['label']

    wsi_ids_test = test_features['wsi_id'].tolist() # To display the WSIs that were misclassified

    random_forest_classifier(x_train, y_train, x_valid, y_valid, x_test, y_test, wsi_ids_test)