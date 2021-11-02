import os
import tensorflow as tf
from config_files.config import Config


class ConfigUmdTest(Config):

    # Paths to files
    FILES_NAME = "affcontext_umd"

    DATA_MASKS_DIR = os.path.join(Config.DATA_DIR, 'cache_UMD', 'GTsegmask_VOC_2012_train')        # for IIt dataset
    DATA_MASKS_DIR_IMGS = os.path.join(Config.DATA_DIR, 'cache_UMD', 'GTsegmask_VOC_2012_train_images')        # for IIt dataset
    WEIGHTS_FILE = os.path.join(Config.ROOT_DIR, "weights_umd", FILES_NAME+".hdf5")

    # General configuration
    IMDB_NAME = "voc_2012_test_umd"         # Available: voc_2012_train_iit, voc_2012_test_iit, voc_2012_train_umd, voc_2012_test_umd
    NUM_CLASSES = 2
    NUM_AFFORDANCE_CLASSES = 8
    AFFORDANCE_LABELS = ["background", "grasp", "cut", "scoop", "contain", "pound", "support", "w-grasp"]

    EVALUATE = True
    VISUALIZE = False

    OBJ_TO_ATTR = tf.constant([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],  # knife
        [1, 1, 0, 0, 0, 0, 0],  # saw
        [1, 1, 0, 0, 0, 0, 0],  # scissors
        [1, 1, 0, 0, 0, 0, 0],  # shears
        [1, 0, 1, 0, 0, 0, 0],  # scoop
        [1, 0, 1, 0, 0, 0, 0],  # spoon
        [1, 0, 1, 0, 0, 0, 0],  # trowel
        [0, 0, 0, 1, 0, 0, 0],  # bowl
        [0, 0, 0, 1, 0, 0, 1],  # cup
        [1, 0, 0, 1, 0, 0, 0],  # ladle
        [1, 0, 0, 1, 0, 0, 1],  # mug
        [0, 0, 0, 1, 0, 0, 1],  # pot
        [1, 0, 0, 0, 0, 1, 0],  # shovel
        [1, 0, 0, 0, 0, 1, 0],  # turner
        [1, 0, 0, 0, 1, 0, 0],  # hammer
        [1, 0, 0, 0, 1, 0, 0],  # mallet
        [1, 0, 0, 0, 1, 0, 0]]  # tenderizer
    )
