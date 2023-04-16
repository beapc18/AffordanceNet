import tensorflow as tf
from tensorflow import keras
from utils import io_utils, data_utils, train_utils, bbox_utils, drawing_utils, eval_utils
from models import affordancenet
import os
import numpy as np
import pickle
import importlib


# Adapt data to tf dataset
def iit_data_to_tf_dataset():
    for roi in roidb:
        final_obj = {
            'image':  roi['image'],     # imagepath
            'image_shape': tf.constant([0., 0.], dtype=tf.float32),  # empty variable for future img shape
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


# Load image and mask data
def load_imgs_and_masks(obj_json):
    if cfg.MASK_REG:
        masks = tf.map_fn(lambda x: data_utils.read_mask(x), obj_json['objects']['mask_path'], dtype=tf.uint8)
        obj_json['objects']['mask'] = masks

    # load the raw data from the file as a string and decode image
    img = tf.io.read_file(obj_json['image'])
    obj_json['image'] = tf.image.decode_jpeg(img, channels=3)
    return obj_json


if __name__ == '__main__':
    keras.backend.clear_session()

    # Read config file
    args = io_utils.handle_args()
    # print('Config file:', args.config_file)
    config = importlib.import_module('config_files.' + args.config_file)
    cfg = config.ConfigIitTest()

    # import backbone
    if cfg.BACKBONE == "mobilenet_v2":
        from models.rpn_mobilenet_v2 import get_model as get_rpn_model
    else:
        from models.rpn_vgg16 import get_model as get_rpn_model

    # Load dataset
    imdb, roidb = data_utils.combined_roidb(cfg.IMDB_NAME, cfg.PROPOSAL_METHOD, cfg.USE_FLIPPED, mode="inference")
    labels = imdb.classes
    total_items = imdb.num_images
    print('{:d} roidb entries'.format(len(roidb)))
    print(imdb.num_classes)
    print(labels)

    # Create tensorflow dataset with the image path and then load the images and masks
    out_types = data_utils.get_data_types()
    test_data = tf.data.Dataset.from_generator(iit_data_to_tf_dataset, output_types=out_types)
    test_data = test_data.map(load_imgs_and_masks, num_parallel_calls=6)
    # print(list(test_data.as_numpy_iterator())[0])

    # Preprocess data
    test_data = test_data.map(
        lambda x: data_utils.preprocessing_iit_dataset_no_resize(x, cfg.MASK_REG))

    data_shapes = data_utils.get_data_shapes(cfg.MASK_REG)
    padding_values = data_utils.get_padding_values(cfg.MASK_REG)
    # drop_remainder -> removes last batch if it's smaller than batch_size
    test_data = test_data.padded_batch(cfg.BATCH_SIZE, padded_shapes=data_shapes, padding_values=padding_values, drop_remainder=True)

    base_anchors = bbox_utils.generate_base_anchors(cfg)
    test_feed = train_utils.iit_generator_inference_no_resize(test_data, cfg)

    # Create models and load weights
    rpn_model, feature_extractor = get_rpn_model(cfg)
    frcnn_model = affordancenet.get_affordance_net_model(feature_extractor, rpn_model, cfg, base_anchors, mode="inference")
    status = frcnn_model.load_weights(cfg.WEIGHTS_FILE, by_name=True)

    # Init statistics for evaluation
    obj_stats = eval_utils.init_stats(labels)
    affordance_stats = eval_utils.init_stats(cfg.AFFORDANCE_LABELS)

    # Save all boxes as in original affordance net [y1, x1, y2, x2, score]
    all_boxes = [[[] for _ in range(total_items)]
                 for _ in range(cfg.NUM_CLASSES)]
    for n in range(total_items):
        for c in range(1, cfg.NUM_CLASSES): #ignore bg class --> [[]]
            all_boxes[c][n] = np.zeros((0,5),dtype=np.float32)


    # Inferece for each image in test set
    for i, image_data in enumerate(test_feed):
        if i == total_items:
            break
        print('Image', i)
        img, image_shape, true_bboxes, true_labels, true_masks, mask_ids = image_data
        print(img)
        pred_bboxes, pred_labels, pred_scores, pred_masks = frcnn_model.predict([img], verbose=1)

        image_shape = tf.squeeze(image_shape, axis=0).numpy().astype(int)

        # Evaluate results
        if cfg.EVALUATE:
            obj_stats, affordance_stats = eval_utils.update_stats(obj_stats, true_bboxes, true_labels, pred_bboxes, pred_labels,
                                                                  pred_scores, cfg.MASK_REG, affordance_stats, true_masks, pred_masks,
                                                                  image_shape[0], image_shape[1], cfg.TEST_NMS_THRESHOLD)

        # Store the final results
        try:
          if cfg.STORE_BBOXES:
              img = tf.squeeze(img, axis=0)
              pred_bboxes = tf.squeeze(pred_bboxes, axis=0)
              pred_labels = tf.squeeze(pred_labels, axis=0)
              pred_scores = tf.squeeze(pred_scores, axis=0)
              true_bboxes = tf.squeeze(true_bboxes, axis=0)
              true_labels = tf.squeeze(true_labels, axis=0)
              # save bboxes for posterior evaluation
              for bbox_index, bbox in enumerate(pred_bboxes):
                  c = int(pred_labels[bbox_index].numpy())
                  denormalized_bboxes = bbox_utils.denormalize_bboxes(bbox, image_shape[0], image_shape[1])
                  y1, x1, y2, x2 = denormalized_bboxes
                  box_score = np.hstack([x1, y1, x2, y2, pred_scores[bbox_index]])
                  all_boxes[c][i] = np.vstack([all_boxes[c][i], box_score])
        except: 
          print("couldnt save boxes of that image")


        # Visualize results
        if cfg.VISUALIZE:
            img = tf.squeeze(img, axis=0)
            pred_bboxes = tf.squeeze(pred_bboxes, axis=0)
            pred_labels = tf.squeeze(pred_labels, axis=0)
            pred_scores = tf.squeeze(pred_scores, axis=0)
            pred_masks = tf.squeeze(pred_masks, axis=0)

            drawing_utils.draw_predictions_with_masks(img, pred_bboxes, pred_labels, pred_scores,
                                           labels, cfg.BATCH_SIZE, cfg.MASK_REG, pred_masks, cfg.AFFORDANCE_LABELS, 'iit')

    # Calculate final FwB score
    if cfg.EVALUATE:
        eval_utils.calculate_mAP(obj_stats)
        if cfg.MASK_REG:
            eval_utils.calculate_f_w_beta_measurement(affordance_stats)

    # Save final file
    if cfg.STORE_BBOXES:
        det_file = os.path.join(cfg.STORE_BBOXES_DIR, 'detections.pkl')
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, 2)
