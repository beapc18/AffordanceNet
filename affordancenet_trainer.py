import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from utils import io_utils, data_utils, train_utils, bbox_utils
from models import affordancenet
import os
import numpy as np
from models import affordancenet_context

import importlib


# Generator to create tensorflow dataset with iit dataset (same structure as pascal dataset load using tfds)
def iit_data_to_tf_dataset():
    for roi in roidb:
        final_obj = {
            'image':  roi['image'],     # imagepath
            'image_shape':  tf.constant([0., 0.], dtype=tf.float32),     # empty variable for future img shape
            'flipped': roi['flipped'],
            'objects': {
                'bbox': tf.constant(roi['boxes'], dtype=tf.float32),
                'label': tf.constant(roi['gt_classes'], dtype=tf.int32),
                'seg_mask_inds': tf.constant(0, dtype=tf.int32),  # keep only mask label
                'mask_path': tf.constant(np.array(['s']), dtype=tf.string),
                'mask':  tf.constant(tf.constant(0.), dtype=tf.float32)
            }
        }
        if cfg.MASK_REG:
            # Create paths to read masks afterwards
            num_bboxes = len(roi['boxes'])
            im_ind = str(roi['seg_mask_inds'][0][0])
            mask_paths = [os.path.join(cfg.DATA_MASKS_DIR_IMGS, im_ind + '_' + str(i) + '_segmask.png') for i in
                          range(1, num_bboxes + 1)]
            final_obj['objects']['seg_mask_inds'] = tf.constant(roi['seg_mask_inds'][:,1], dtype=tf.int32)    # keep only mask label
            final_obj['objects']['mask_path'] = tf.constant(np.array(mask_paths), dtype=tf.string)
        yield final_obj


# Load images in the dataset from the paths
def load_imgs_and_masks(obj_json):
    if cfg.MASK_REG:
        masks = tf.map_fn(lambda x: data_utils.read_mask(x), obj_json['objects']['mask_path'], dtype=tf.uint8)
        if obj_json['flipped']:
            masks = tf.image.flip_left_right(masks)
        obj_json['objects']['mask'] = masks

    # load the raw data from the file as a string and decode image
    img = tf.io.read_file(obj_json['image'])
    img = tf.image.decode_jpeg(img, channels=3)
    if obj_json['flipped']:
        img = tf.image.flip_left_right(img)
    obj_json['image'] = img
    return obj_json


if __name__ == '__main__':
    keras.backend.clear_session()

    # Read config file
    args = io_utils.handle_args()
    # print('Config file:', args.config_file)
    config = importlib.import_module('config_files.'+args.config_file)
    cfg = config.ConfigIitTrain()

    io_utils.is_valid_backbone(cfg.BACKBONE)

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    if cfg.BACKBONE == "mobilenet_v2":
        from models.rpn_mobilenet_v2 import get_model as get_rpn_model
    else:
        from models.rpn_vgg16 import get_model as get_rpn_model

    # Load dataset
    imdb, roidb = data_utils.combined_roidb(cfg.IMDB_NAME, cfg.PROPOSAL_METHOD, cfg.USE_FLIPPED, mode="training")
    print('{:d} roidb entries'.format(len(roidb)))
    print(imdb.num_classes)

    # Create tensorflow dataset with the image path and then load the images and masks
    out_types = data_utils.get_data_types_training()
    train_data = tf.data.Dataset.from_generator(iit_data_to_tf_dataset, output_types=out_types)

    # shuffle data after every epoch
    train_total_items = len(roidb)
    train_data = train_data.shuffle(train_total_items, reshuffle_each_iteration=True)

    train_data = train_data.map(load_imgs_and_masks, num_parallel_calls=6)
    base_anchors = bbox_utils.generate_base_anchors(cfg)

    # Preprocess data
    train_data = train_data.map(
        lambda x: data_utils.preprocessing_iit_dataset_no_resize(x, cfg.MASK_REG))

    data_shapes = data_utils.get_data_shapes(cfg.MASK_REG)
    padding_values = data_utils.get_padding_values(cfg.MASK_REG)
    # drop_remainder -> removes last batch if it's smaller than batch_size
    train_data = train_data.padded_batch(cfg.BATCH_SIZE, padded_shapes=data_shapes, padding_values=padding_values,
                                         drop_remainder=True)
    train_feed = train_utils.iit_generator_no_resize(train_data, cfg, base_anchors)

    # Create models
    rpn_model, feature_extractor = get_rpn_model(cfg)
    frcnn_model = affordancenet.get_affordance_net_model(feature_extractor, rpn_model, cfg, base_anchors)

    step_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        cfg.LEARNING_RATE, cfg.DECAY_STEPS, cfg.DECAY_RATE, staircase=True, name='lr_decay'
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=step_decay, momentum=cfg.MOMENTUM, name="SGD")
    frcnn_model.compile(optimizer=optimizer, loss=[None] * len(frcnn_model.output), run_eagerly=cfg.RUN_EAGERLY)
    affordancenet.init_model_no_resize(frcnn_model, cfg)

    frcnn_model.summary(line_length=200)
    keras.utils.plot_model(frcnn_model, show_shapes=True)

    # Load weights
    if cfg.USE_WEIGHTS:
        frcnn_model.load_weights(cfg.WEIGHTS_FILE)

    step_size_train = train_utils.get_step_size(train_total_items, cfg.BATCH_SIZE)

    checkpoint_callback = ModelCheckpoint(cfg.CHECKPOINT_DIR, save_weights_only=True, include_optimizer=False)
    tensorboard_callback = affordancenet_context.LRTensorBoard(log_dir=cfg.LOG_DIR)

    frcnn_model.fit(train_feed,
                    steps_per_epoch=step_size_train,
                    epochs=cfg.EPOCHS,
                    callbacks=[checkpoint_callback, tensorboard_callback])

    # Save weigths at the end
    frcnn_model.save_weights(cfg.WEIGHTS_FILE)

    # # Define our metrics for TensorBoard
    # train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    # train_summary_writer = tf.summary.create_file_writer(cfg.LOG_DIR)
    #
    # epoch = -1
    # accumulate_grads = []
    # accumulate_vars = []
    # # iterate over the generator
    # for (batch, inputs) in enumerate(train_feed):
    #     # new epoch
    #     if batch % step_size_train == 0:
    #         if epoch >= cfg.EPOCHS:
    #             break
    #
    #         if epoch >= 0:
    #             # write results for tensorboard at the end of the epoch
    #             with train_summary_writer.as_default():
    #                 tf.summary.scalar('epoch_loss', train_loss.result(), step=epoch)
    #             print('epoch', epoch, '-', train_loss.result())
    #
    #         train_loss.reset_states()  # reset states at the beginning of the epoch
    #         epoch += 1
    #         print('epoch', epoch)
    #
    #
    #     (batch_imgs, batch_img_shape, batch_gt_bboxes, batch_gt_labels, batch_rpn_bboxes, batch_rpn_labels, batch_gt_mask,
    #      batch_gt_seg_mask_inds), y = inputs
    #     with tf.GradientTape() as tape:
    #         rpn_reg_predictions, rpn_cls_predictions, frcnn_reg_predictions, frcnn_cls_predictions, mask_prob_output = frcnn_model(
    #             [batch_imgs, batch_img_shape, batch_gt_bboxes, batch_gt_labels, batch_rpn_bboxes, batch_rpn_labels, batch_gt_mask,
    #              batch_gt_seg_mask_inds], training=True)
    #
    #         loss_value = frcnn_model.compiled_loss(
    #             y,
    #             (rpn_reg_predictions, rpn_cls_predictions, frcnn_reg_predictions, frcnn_cls_predictions,
    #              mask_prob_output),
    #             None, regularization_losses=frcnn_model.losses)
    #
    #     grads = tape.gradient(loss_value, frcnn_model.trainable_variables)
    #     grads = optimizer._clip_gradients(grads)    # gradient clipping to avoid exploding gradients
    #     accumulate_grads.append(grads)
    #     accumulate_vars.append(frcnn_model.trainable_variables)
    #     train_loss(loss_value)  # gather loss info for tensorboard
    #     print('iteration', batch % step_size_train, 'of', step_size_train, '- loss:', loss_value)
    #
    #     # backpropagate only if the number of steps = batch size
    #     if batch % cfg.ITER_SIZE == 0:
    #         accumulate_grads = np.asarray(accumulate_grads).flatten().tolist()
    #         accumulate_vars = np.asarray(accumulate_vars).flatten().tolist()
    #
    #         frcnn_model.optimizer.apply_gradients(zip(accumulate_grads, accumulate_vars))
    #         accumulate_grads = []
    #         accumulate_vars = []


