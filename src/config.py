import os

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PARENT_PATH = os.getcwd()

FILES_PATH = os.path.join(PARENT_PATH, 'files')
LABELS_PATH = os.path.join(FILES_PATH, 'Labels.xlsx')
WSI_FILENAMES = os.path.join(FILES_PATH, 'lgg_wsi_filenames.xlsx')
STAIN_NORMALIZATION_REF = os.path.join(FILES_PATH, 'stain_normalization_ref.png')

# Paths of the WSIs directories
WSI_PARENT_PATH = os.path.join(PARENT_PATH, 'WSI')
LGG_WSI_PATH = os.path.join(WSI_PARENT_PATH, 'LGG')
WSI_BLOCKS_PATH = os.path.join(LGG_WSI_PATH, 'blocks')
ANNOTATIONS_PATH = os.path.join(LGG_WSI_PATH, 'annotations')
INFER_WSI_PATH = os.path.join(WSI_PARENT_PATH, 'inference')

PATCHES_PATH = os.path.join(PARENT_PATH, 'patches')
VALIDATE_PATCHES_PATH = os.path.join(FILES_PATH, 'validate_patches')

# Files generated in patch_extraction
TRAIN_PATCHES = os.path.join(PATCHES_PATH, 'train_patches')
VALID_PATCHES = os.path.join(PATCHES_PATH, 'valid_patches')
TEST_PATCHES = os.path.join(PATCHES_PATH, 'test_patches')
INFER_PATCHES = os.path.join(PATCHES_PATH, 'inference_patches') # Generated in infer_wsi.py
COORDS_FILE_NAME = os.path.join(FILES_PATH, 'patches_coords.xlsx')
COORDS_INFER_FILE_NAME = os.path.join(FILES_PATH, 'patches_coords_infer.xlsx') # Generated in infer_wsi.py

# Files generated in classification_patches
dropout_rate = 0.5

WEIGHTS_PATH = os.path.join(FILES_PATH, 'weights.pth') # Update for transfer learning from another weights file
BEST_MODEL_PATH = os.path.join(FILES_PATH, 'best_model.pth')
BEST_MODEL_VALIDATE_PATCHES_PATH = os.path.join(FILES_PATH, 'best_model_validate_patches.pth')

TRAIN_PREDICTIONS = os.path.join(FILES_PATH, 'train_predictions.xlsx')
TRAIN_FEATURES = os.path.join(FILES_PATH, 'train_features.pkl')

VALID_PREDICTIONS = os.path.join(FILES_PATH, 'valid_predictions.xlsx')
VALID_FEATURES = os.path.join(FILES_PATH, 'valid_features.pkl')

TEST_PREDICTIONS = os.path.join(FILES_PATH, 'test_predictions.xlsx')
TEST_FEATURES = os.path.join(FILES_PATH, 'test_features.pkl')

INFER_PREDICTIONS = os.path.join(FILES_PATH, 'infer_predictions.xlsx')
INFER_FEATURES = os.path.join(FILES_PATH, 'infer_features.pkl')

ROC_PLOT_PATH = os.path.join(FILES_PATH, 'roc_plot.png')

# Files generated in aggregation_features
TRAIN_AGG_FV = os.path.join(FILES_PATH, 'train_feature_vector.pkl')
VALID_AGG_FV = os.path.join(FILES_PATH, 'valid_feature_vector.pkl')
TEST_AGG_FV = os.path.join(FILES_PATH, 'test_feature_vector.pkl')

TRAIN_WSI_CLASSIFICATION = os.path.join(FILES_PATH, 'train_WSI_classification.xlsx')
VALID_WSI_CLASSIFICATION = os.path.join(FILES_PATH, 'valid_WSI_classification.xlsx')
TEST_WSI_CLASSIFICATION = os.path.join(FILES_PATH, 'test_WSI_classification.xlsx')

# Files generated in classification_wsi
RANDOM_FOREST_MODEL = os.path.join(FILES_PATH, 'best_random_forest.pkl')
SVM_MODEL = os.path.join(FILES_PATH, 'best_svm.pkl')
XGBOOST_MODEL = os.path.join(FILES_PATH, 'best_xgboost.pkl')

KMEANS_CLUSTER_PATH = os.path.join(FILES_PATH, 'kmeans_cluster.png')
DBSCAN_CLUSTER_PATH = os.path.join(FILES_PATH, 'dbscan_cluster.png')
PCA_TRAIN_LABEL_PATH = os.path.join(FILES_PATH, 'pca_train.png')
PCA_TEST_LABEL_PATH = os.path.join(FILES_PATH, 'pca_test.png')
UMAP_TRAIN_LABEL_PATH = os.path.join(FILES_PATH, 'umap_train.png')
UMAP_TEST_LABEL_PATH = os.path.join(FILES_PATH, 'umap_test.png')
CONFUSION_MATRIX_PATH = os.path.join(FILES_PATH, 'confusion_matrix.png')

# Files generated in infer_wsi
INFERENCE_PREDICTIONS = os.path.join(FILES_PATH, 'inference_predictions.xlsx')
INFERENCE_FEATURES = os.path.join(FILES_PATH, 'inference_features.pkl')

# Files generated in aggregation_multimodal_features
TRAIN_AGG_MULTI_FV = os.path.join(FILES_PATH, 'train_feature_vector_mulit.pkl')
VALID_AGG_MULTI_FV = os.path.join(FILES_PATH, 'valid_feature_vector_mulit.pkl')
TEST_AGG_MULTI_FV = os.path.join(FILES_PATH, 'test_feature_vector_mulit.pkl')

# Files generated in visualize_predictions
HEATMAPS_PATH =  os.path.join(FILES_PATH, 'heatmaps')

# Files generated in misc
STATS_PLOT_PATH = os.path.join(FILES_PATH, 'stats.png')

# ---------------------------------------------------------------------------
# LISTS
# ---------------------------------------------------------------------------
CLASSES = ['BRAF fusion', 'BRAF SNV', 'FGFR altered']
ALL_PATCHES = [TRAIN_PATCHES, TEST_PATCHES]

# ---------------------------------------------------------------------------
# NUMERICAL VALUES
# ---------------------------------------------------------------------------
PATCH_SIZE = 512
TARGET_MAGNIFICATION = 10
MAX_NUM_PATCHES = 5000 
FEATURE_VECTOR_SIZE = 128


# ---------------------------------------------------------------------------
# EXTENSION TO PATHS
# ---------------------------------------------------------------------------
def add_extension(path, additional_ext):
    dir = os.path.dirname(path)
    file_name, extension = os.path.splitext(os.path.basename(path))[0], os.path.splitext(os.path.basename(path))[1]
    return os.path.join(dir, file_name + str(additional_ext) + extension)
