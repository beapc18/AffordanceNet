import os
import tensorflow as tf
from config_files.config import Config


class ConfigUmdTrain(Config):

    # Paths to files
    FILES_NAME = "affcontext_new"

    CHECKPOINT_DIR = os.path.join(Config.ROOT_DIR, "checkpoints", FILES_NAME, "ckpt")
    LOG_DIR = os.path.join(Config.ROOT_DIR, "logs", FILES_NAME)
    DATA_MASKS_DIR = os.path.join(Config.DATA_DIR, 'cache_UMD', 'GTsegmask_VOC_2012_train')
    DATA_MASKS_DIR_IMGS = os.path.join(Config.DATA_DIR, 'cache_UMD', 'GTsegmask_VOC_2012_train_images')

    # General configuration
    IMDB_NAME = "voc_2012_train_umd"        # Available: voc_2012_train_iit, voc_2012_test_iit, voc_2012_train_umd, voc_2012_test_umd
    NUM_CLASSES = 2                         # Available: 11 for IIT, 2 for UMD
    NUM_AFFORDANCE_CLASSES = 8              # Available: 10 for IIT, 8 for UMD
    AFFORDANCE_LABELS = ["background", "grasp", "cut", "scoop", "contain", "pound", "support", "w-grasp"]

    EPOCHS = 10
    USE_FLIPPED = True      # Add flipped images for training only
    TRAIN_ROIS_PER_IMAGE = 36
    ROI_POSITIVE_RATIO = 0.25

    # LOSS_NAMES = Config.LOSS_NAMES.append("attr_loss")
    # LOSS_WEIGHTS = Config.LOSS_WEIGHTS.append(Config.AFFORDANCE_ATTR_LOSS_WEIGHT)
    LOSS_NAMES = ["rpn_reg_loss", "rpn_cls_loss", "reg_loss", "cls_loss", "mask_loss", "attr_loss"]
    LOSS_WEIGHTS = [Config.RPN_BBOX_LOSS_WEIGHT, Config.RPN_CLASS_LOSS_WEIGHT, Config.AFFORDANCE_BBOX_LOSS_WEIGHT,
                    Config.AFFORDANCE_CLASS_LOSS_WEIGHT, Config.AFFORDANCE_MASK_LOSS_WEIGHT, Config.AFFORDANCE_ATTR_LOSS_WEIGHT]

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
