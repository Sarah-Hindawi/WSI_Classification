import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, log_loss

import config as c
import classification_wsi

def classify_multimodal_features(x_train, y_train, x_valid, y_valid, x_test, y_test):

    catboost_params = {
    "learning_rate": 0.003,
    "depth": 5,
    "l2_leaf_reg": 10,
    "iterations": 2000,
    "loss_function": 'Logloss',
    "verbose": 200
    }

    clf = CatBoostClassifier(**catboost_params)

    clf.fit(x_train[:len(x_train)], y_train, eval_set=(x_train[len(x_train):len(x_train)+len(x_valid)], y_valid))

    valid_predictions = clf.predict_proba(valid_predictions[len(x_train):len(x_train)+len(x_valid)])
    print(f"Log loss on validation set: {log_loss(y_valid, valid_predictions):.3f}")
 
    # clf.save_model("catboost_model.cbm")

def train_two_models(x_train_wsi, x_train_clinical, y_train, x_valid_wsi, x_valid_clinical, y_valid):
    
    # Train XGBoost on WSI features
    xgb_wsi = xgb.XGBClassifier()
    xgb_wsi.fit(x_train_wsi, y_train)
    valid_wsi_preds = xgb_wsi.predict_proba(x_valid_wsi)

    # Train XGBoost on clinical features
    xgb_clinical = xgb.XGBClassifier()
    xgb_clinical.fit(x_train_clinical, y_train)
    valid_clinical_preds = xgb_clinical.predict_proba(x_valid_clinical)

    return xgb_wsi, xgb_clinical, valid_wsi_preds, valid_clinical_preds


def ensemble_predictions(wsi_preds, clinical_preds, alpha=0.5):
    return alpha * wsi_preds + (1 - alpha) * clinical_preds


def find_best_alpha(wsi_preds, clinical_preds, y_valid):
    alphas = np.linspace(0, 1, 11)  # Trying alphas from 0 to 1 with a step of 0.1
    best_log_loss = float('inf')
    best_alpha = None

    for alpha in alphas:
        combined_preds = ensemble_predictions(wsi_preds, clinical_preds, alpha)
        current_log_loss = log_loss(y_valid, combined_preds)

        if current_log_loss < best_log_loss:
            best_log_loss = current_log_loss
            best_alpha = alpha

    return best_alpha

def xgboost_classifier_ensemble(x_train_wsi, x_train_clinical, y_train, x_valid_wsi, x_valid_clinical, y_valid, x_test_wsi, x_test_clinical, y_test):
    xgb_wsi, xgb_clinical, valid_wsi_preds, valid_clinical_preds = train_two_models(x_train_wsi, x_train_clinical, y_train, x_valid_wsi, x_valid_clinical, y_valid)

    best_alpha = find_best_alpha(valid_wsi_preds, valid_clinical_preds, y_valid)

    test_wsi_preds = xgb_wsi.predict_proba(x_test_wsi)
    test_clinical_preds = xgb_clinical.predict_proba(x_test_clinical)
    final_test_preds = ensemble_predictions(test_wsi_preds, test_clinical_preds, best_alpha)
    
    # Determining the predicted class label for multi-class
    test_preds_label = np.argmax(final_test_preds, axis=1)
    test_accuracy = accuracy_score(y_test, test_preds_label)

    print(f"Best alpha for ensemble: {best_alpha}")
    print(f"Accuracy on test set: {test_accuracy}")

def main():

    train_fv = pd.DataFrame(pd.read_pickle(c.TRAIN_AGG_MULTI_FV))
    valid_fv = pd.DataFrame(pd.read_pickle(c.VALID_AGG_MULTI_FV))
    test_fv = pd.DataFrame(pd.read_pickle(c.TEST_AGG_MULTI_FV))

    # Flatten and concatenate all feature vectors
    x_train = np.concatenate(train_fv['features'].tolist()).reshape(-1, 542)
    x_valid = np.concatenate(valid_fv['features'].tolist()).reshape(-1, 542)
    x_test = np.concatenate(test_fv['features'].tolist()).reshape(-1, 542)

    y_train = train_fv['label']
    y_valid = valid_fv['label']
    y_test = test_fv['label']

    classification_wsi.xgboost_classifier(x_train, y_train, x_valid, y_valid, x_test, y_test)

if __name__ == '__main__':
    main()