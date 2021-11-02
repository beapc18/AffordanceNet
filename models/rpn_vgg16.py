import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.regularizers as KR

def get_model(cfg):
    """Generating rpn model for given hyper params.
    inputs:
        hyper_params = dictionary

    outputs:
        rpn_model = tf.keras.model
        feature_extractor = feature extractor layer from the base model
    """
    base_model = VGG16(include_top=False, input_shape=(None, None, 3))

    # no train first layers
    base_model.layers[1].trainable = False
    base_model.layers[2].trainable = False
    base_model.layers[3].trainable = False
    base_model.layers[4].trainable = False

    feature_extractor = base_model.get_layer("block5_conv3")
    l2_reg = KR.l2(cfg.WEIGHT_DECAY)

    output = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_regularizer=l2_reg, name="rpn_conv")(feature_extractor.output)
    rpn_cls_output = Conv2D(cfg.ANCHOR_COUNT, (1, 1), activation="sigmoid", kernel_regularizer=l2_reg, name="rpn_cls")(output)
    rpn_reg_output = Conv2D(cfg.ANCHOR_COUNT * 4, (1, 1), activation="linear", kernel_regularizer=l2_reg, name="rpn_reg")(output)
    rpn_model = Model(inputs=base_model.input, outputs=[rpn_reg_output, rpn_cls_output])
    return rpn_model, feature_extractor

def init_model(model):
    """Initializing model with dummy data for load weights with optimizer state and also graph construction.
    inputs:
        model = tf.keras.model
    """
    model(tf.random.uniform((1, 500, 500, 3)))
