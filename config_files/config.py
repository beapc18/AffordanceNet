import os
import numpy as np

# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override the configurations you need to change.


class Config(object):
    # Root directory of project
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Paths to files
    FILES_NAME = ""
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    VOC_DOWNLOAD_DIR = os.path.join(DATA_DIR, 'voc_download')
    STORE_BBOXES_DIR = os.path.join(ROOT_DIR, 'detections')
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints", FILES_NAME, "ckpt")
    LOG_DIR = os.path.join(ROOT_DIR, "logs", FILES_NAME)
    WEIGHTS_FILE = os.path.join(ROOT_DIR, "weights", FILES_NAME+".hdf5")
    # DATA_MASKS_DIR = os.path.join(DATA_DIR, 'cache', 'GTsegmask_VOC_2012_train')        # for IIt dataset
    # DATA_MASKS_DIR_IMGS = os.path.join(DATA_DIR, 'cache', 'GTsegmask_VOC_2012_train_images')        # for IIt dataset
    # DATA_MASKS_DIR = os.path.join(DATA_IIT_DIR, 'cache_UMD', 'GTsegmask_VOC_2012_train')  # for UMD dataset
    # DATA_MASKS_DIR_IMGS = os.path.join(DATA_IIT_DIR, 'cache_UMD', 'GTsegmask_VOC_2012_train_images')  # for UMD dataset

    # General configuration
    DATASET = "voc/2007"                    # Available: voc/2007 (for iit and umd), coco/2017
    TRAINING_SET = "train+validation"
    VAL_SET = "test"

    # Depend on dataset
    IMDB_NAME = ""                          # Available: voc_2012_train_iit, voc_2012_test_iit, voc_2012_train_umd, voc_2012_test_umd
    NUM_CLASSES = 0                         # Available: 11 for IIT, 2 for UMD
    NUM_AFFORDANCE_CLASSES = 0              # Available: 10 for IIT, 8 for UMD
    AFFORDANCE_LABELS = []

    ITER_SIZE = 2           # Gradient accumulation
    BATCH_SIZE = 1
    EPOCHS = 0              # Total_iterations / it_per_epoch (200K/1250)
    USE_WEIGHTS = False
    BACKBONE = 'vgg16'      # vgg16 or mobilenet_v2
    EVALUATE = False
    VISUALIZE = False
    STORE_BBOXES = False
    MASK_REG = True         # Include masks
    RUN_EAGERLY = False
    USE_FLIPPED = False      # Add flipped images for training only
    TRAIN_MASK_SIZE = 224

    # Training
    LEARNING_RATE = 0.001
    DECAY_STEPS = 150000
    DECAY_RATE = 0.1

    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    PROPOSAL_METHOD = "gt"

    IMG_SIZE_WIDTH = 500
    IMG_SIZE_HEIGHT = 500
    ANCHOR_RATIOS = [1., 2., 1. / 2.]
    ANCHOR_SCALES = [32, 64, 128, 256, 512]
    FEATURE_MAP_SHAPE = 31
    ANCHOR_COUNT = len(ANCHOR_RATIOS) * len(ANCHOR_SCALES)

    # Train RPN
    RPN_POSITIVE_OVERLAP = 0.7  # select if an anchor box is a good fg box
    RPN_NEGATIVE_OVERLAP = 0.3  # if the max overlap of a anchor from a ground truth box is lower than this thershold, it is marked as background. Boxes whose overlap is > than RPN_NEGATIVE_OVERLAP but < RPN_POSITIVE_OVERLAP are marked “don’t care”
    RPN_BATCHSIZE = 256         # total of bg and fg anchors  --> for targets
    RPN_FG_FRACTION = 0.5       # fraction of the batch size that is fg anchors (128)
    VARIANCES = np.array([0.1, 0.1, 0.2, 0.2])

    # Parameters for proposal layer
    NMS_IOU_THRESHOLD = 0.7         # NMS threshold used on RPN proposals
    PRE_NMS_TOPN = 12000            # Number of top scoring boxes to keep before apply NMS to RPN proposals
    TRAIN_NMS_TOPN= 2000            # Number of top scoring boxes to keep after applying NMS to RPN proposals
    TEST_PRE_NMS_TOPN = 6000
    TEST_NMS_TOPN = 1000

    # Parameters for proposal target layer
    TRAIN_FG_THRES = 0.5  # Threshold for IoU positive ROIs
    TRAIN_BG_THRESH_LO = 0.1  # Threshold for IoU negative ROIs
    TRAIN_BG_THRESH_HI = 0.5  # Threshold for IoU negative ROIs
    TRAIN_ROIS_PER_IMAGE = 32  # Number of ROIs per image to feed to classifier/mask heads =TRAIN.BATCH_SIZE
    ROI_POSITIVE_RATIO = 0.3  # Percent of positive ROIs used to train classifier/mask heads =TRAIN.FG_FRACTION

    # Bounding box refinement standard deviation for RPN and final detections. = VARIANCE
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # ROI align layer
    POOL_SIZE = 7

    # Weights for each loss
    LOSS_NAMES = ["rpn_reg_loss", "rpn_cls_loss", "reg_loss", "cls_loss", "mask_loss"] #, "attr_loss"]
    RPN_BBOX_LOSS_WEIGHT = 1.0
    RPN_CLASS_LOSS_WEIGHT = 1.0
    AFFORDANCE_BBOX_LOSS_WEIGHT = 2.0
    AFFORDANCE_CLASS_LOSS_WEIGHT = 3.0
    AFFORDANCE_MASK_LOSS_WEIGHT = 3.0
    AFFORDANCE_ATTR_LOSS_WEIGHT = 1.0
    LOSS_WEIGHTS = [RPN_BBOX_LOSS_WEIGHT, RPN_CLASS_LOSS_WEIGHT, AFFORDANCE_BBOX_LOSS_WEIGHT,
                    AFFORDANCE_CLASS_LOSS_WEIGHT, AFFORDANCE_MASK_LOSS_WEIGHT] #, AFFORDANCE_ATTR_LOSS_WEIGHT]

    # TEST
    TEST_NMS_THRESHOLD = 0.3  # 0.5 before for nonmaximum suppression
    SCORE_THRESHOLD = 0.9
    MAX_PER_IMAGE = 100