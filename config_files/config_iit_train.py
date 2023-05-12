import os
from config_files.config import Config


class ConfigIitTrain(Config):

    # Paths to files
    FILES_NAME = "affordancenet_new"

    CHECKPOINT_DIR = os.path.join(Config.ROOT_DIR, "checkpoints", FILES_NAME, "ckpt")
    LOG_DIR = os.path.join(Config.ROOT_DIR, "logs", FILES_NAME)
    DATA_MASKS_DIR = os.path.join(Config.DATA_DIR, 'cache', 'GTsegmask_VOC_2012_train')        # for IIt dataset
    DATA_MASKS_DIR_IMGS = os.path.join(Config.DATA_DIR, 'cache', 'GTsegmask_VOC_2012_train_images')        # for IIt dataset

    # General configuration
    IMDB_NAME = "voc_2012_train_iit"        # Available: voc_2012_train_iit, voc_2012_test_iit, voc_2012_train_umd, voc_2012_test_umd
    NUM_CLASSES = 11                        # Available: 11 for IIT, 2 for UMD
    NUM_AFFORDANCE_CLASSES = 10             # Available: 10 for IIT, 8 for UMD
    AFFORDANCE_LABELS = ["background", "contain", "cut", "display", "engine", "grasp", "hit", "pound", "support", "w-grasp"]

    EPOCHS = 4             # 25+8
    USE_FLIPPED = True      # Add flipped images for training only