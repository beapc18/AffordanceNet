import os
from config_files.config import Config


class ConfigIitTest(Config):

    # Paths to files
    FILES_NAME = "affordancenet_iit"

    DATA_MASKS_DIR = os.path.join(Config.DATA_DIR, 'cache', 'GTsegmask_VOC_2012_train')        # for IIt dataset
    DATA_MASKS_DIR_IMGS = os.path.join(Config.DATA_DIR, 'cache', 'GTsegmask_VOC_2012_train_images')        # for IIt dataset
    WEIGHTS_FILE = os.path.join(Config.ROOT_DIR, "weights", FILES_NAME+".hdf5")

    # General configuration
    IMDB_NAME = "voc_2012_test_iit"         # Available: voc_2012_train_iit, voc_2012_test_iit, voc_2012_train_umd, voc_2012_test_umd
    NUM_CLASSES = 11                        # Available: 11 IIT dataset,
    NUM_AFFORDANCE_CLASSES = 10             # Available: 10 IIT dataset,
    AFFORDANCE_LABELS = ["background", "contain", "cut", "display", "engine", "grasp", "hit", "pound", "support", "w-grasp"]

    EVALUATE = True
    VISUALIZE = False
    STORE_BBOXES = True
