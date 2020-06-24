# import the necessary packages
import os

# dataset confugirations
# base path to the input dataset
BASE_PATH = "dataset"

# ==============================================================
# chestXray14 dataset configurations
CHESTXRAY14_PATH = os.path.sep.join([BASE_PATH, "chest_xrays"])
CHESTXRAY14_METADATA_PATH = os.path.sep.join(
    [CHESTXRAY14_PATH, "Data_Entry_2017.csv"])
CHESTXRAY14_IMAGES_BASE_PATH = os.path.sep.join(
    [CHESTXRAY14_PATH, "images", "*"])
CHESTXRAY14_COLS = ['Image Index', 'Image Path', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
                    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

# ==============================================================
# TB_chest_xrays dataset configurations
TB_COLS = ['Image Index', 'Image Path', 'Tuberculosis']

# TB_shenzen dataset configurations
TB_SHENZHEN_PATH = os.path.sep.join(
    [BASE_PATH, "tuberculosis-chest-xrays-shenzhen"])
TB_SHENZHEN_METADATA_PATH = os.path.sep.join(
    [TB_SHENZHEN_PATH, "shenzhen_metadata.csv"])
TB_SHENZHEN_IMAGES_BASE_PATH = os.path.sep.join([TB_SHENZHEN_PATH, "images"])

# TB_montgomery dataset configuration
TB_MONTGOMERY_PATH = os.path.sep.join(
    [BASE_PATH, "tuberculosis-chest-xrays-shenzhen"])
TB_MONTGOMERY_METADATA_PATH = os.path.sep.join(
    [TB_MONTGOMERY_PATH, "montgomery_metadata.csv"])
TB_MONTGOMERY_IMAGES_BASE_PATH = os.path.sep.join([TB_SHENZHEN_PATH, "images"])

# ==============================================================
# chest_xrays15
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis',
               'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'Tuberculosis']

# ==============================================================
# training, validation, and testing images paths
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# training, validation, and testing metadata paths
TRAIN_METADATA_PATH = os.path.sep.join([BASE_PATH, "train.csv"])
VAL_METADATA_PATH = os.path.sep.join([BASE_PATH, "validation.csv"])
TEST_METADATA_PATH = os.path.sep.join([BASE_PATH, "test.csv"])

# train, val, and test split ratios
TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.05
TEST_SPLIT = 0.05

# ==============================================================
# basic training parametries
EPOCHS = 100
BATCH_SIZE = 32
INTIAL_LR = 1e-3
MIN_LR = 1e-8

# ==============================================================
# output configurations
OUTPUT_PATH = "output"
MODEL_PATH = os.path.sep.join([OUTPUT_PATH, "models", "LuNet.h5"])
# MODEL_ARCHITECTURE_PATH = os.path.sep.join([OUTPUT_PATH, "models", "LuNet_architecture.json"])
# MODEL_WEIGHT_PATH = os.path.sep.join([OUTPUT_PATH, "models", "LuNet_weights.h5"])
LOG_DIR = os.path.sep.join([OUTPUT_PATH, "logs"])
