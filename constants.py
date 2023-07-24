import os

counter = '2' # Update it when creating a differnet version of the same files instead of overwritting them
# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
OPENSLIDE_PATH = r'C:\Users\sarah\Documents\openslide-win64-20230414\bin'
PARENT_PATH = os.getcwd()
FILES_PATH = '../../WSI Project/WSI_Classification_local/Code/files'
LABELS_PATH = os.path.join(FILES_PATH, "Labels.xlsx")
TRAIN_PATH = 'train_patches' + counter
VALID_PATH = 'valid_patches' + counter
TEST_PATH = 'test_patches' + counter
COORDS_FILE_NAME = 'patches_coords'+counter+'.xlsx'
WSI_PARENT_PATH = os.path.join(PARENT_PATH, 'WSI')
LGG_PATH = os.path.join(WSI_PARENT_PATH, 'LGG')
HGG_PATH = os.path.join(WSI_PARENT_PATH, 'PHGG')
NORMAL_PATH = os.path.join(WSI_PARENT_PATH, 'normal_brain')
WSI_PATHS = [LGG_PATH, HGG_PATH]
TRAIN_PREDICTIONS = os.path.join(FILES_PATH, 'train_predictions'+counter+'.xlsx')
TRAIN_FEATURES = os.path.join(FILES_PATH, 'train_features'+counter+'.pkl')
VALID_PREDICTIONS = os.path.join(FILES_PATH, 'valid_predictions'+counter+'.xlsx')
VALID_FEATURES = os.path.join(FILES_PATH, 'valid_features'+counter+'.pkl')
TEST_PREDICTIONS = os.path.join(FILES_PATH, 'test_predictions'+counter+'.xlsx')
TEST_FEATURES = os.path.join(FILES_PATH, 'test_features'+counter+'.pkl')
BEST_MODEL_PATH = os.path.join(FILES_PATH, 'best_model'+counter+'.pth')


# ---------------------------------------------------------------------------
# LISTS
# ---------------------------------------------------------------------------
CLASSES = ['HGG', 'LGG']


# ---------------------------------------------------------------------------
# NUMERICAL VALUES
# ---------------------------------------------------------------------------
NUM_CLASSES = 2
PATCH_SIZE = 224
TARGET_MAGNIFICATION = 20
BASE_MAGNIFICATION = 20.0  # base_magnification (float): The base magnification level of the WSIs.
NUM_PATCHES = 200
