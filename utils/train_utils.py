import tensorflow as tf
import math
from utils import bbox_utils
from tensorflow import keras
import tensorflow.keras.losses as KLoss


def get_step_size(total_items, batch_size):
    """Get step size for given total item size and batch size.
    inputs:
        total_items = number of total items
        batch_size = number of batch size during training or validation
    outputs:
        step_size = number of step size for model training
    """
    return math.floor(total_items / batch_size)


def randomly_select_xyz_mask(mask, select_xyz):
    """Selecting x, y, z number of True elements for corresponding batch and replacing others to False
    inputs:
        mask = (batch_size, [m_bool_value])
        select_xyz = ([x_y_z_number_for_corresponding_batch])
            example = tf.constant([128, 50, 42], dtype=tf.int32)
    outputs:
        selected_valid_mask = (batch_size, [m_bool_value])
    """
    maxval = tf.reduce_max(select_xyz) * 10
    random_mask = tf.random.uniform(tf.shape(mask), minval=1, maxval=maxval, dtype=tf.int32)
    multiplied_mask = tf.cast(mask, tf.int32) * random_mask
    sorted_mask = tf.argsort(multiplied_mask, direction="DESCENDING")
    sorted_mask_indices = tf.argsort(sorted_mask)
    selected_mask = tf.less(sorted_mask_indices, tf.expand_dims(select_xyz, 1))
    return tf.logical_and(mask, selected_mask)


def iit_generator(dataset, anchors, cfg):
    """Tensorflow data generator for fit method, yielding inputs and outputs.
        inputs:
            dataset = tf.data.Dataset, PaddedBatchDataset
            anchors = (total_anchors, [y1, x1, y2, x2])
                these values in normalized format between [0, 1]
            hyper_params = dictionary

        outputs:
            yield inputs, outputs
        """
    while True:
        for image_data in dataset:
            if cfg.MASK_REG:
                img, gt_boxes, gt_labels, gt_mask, gt_seg_mask_inds = image_data
            else:
                img, gt_boxes, gt_labels = image_data

            bbox_deltas, bbox_labels = calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, cfg)

            if cfg.MASK_REG:
                yield (img, gt_boxes, gt_labels, bbox_deltas, bbox_labels, gt_mask, gt_seg_mask_inds), \
                      (tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32),
                       tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32),
                       tf.constant(0., dtype=tf.float32))
            else:
                yield (img, gt_boxes, gt_labels, bbox_deltas, bbox_labels), \
                      (tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32),
                       tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32))


def iit_generator_inference_no_resize(dataset, cfg):
    """Tensorflow data generator for fit method, yielding inputs and outputs.
        inputs:
            dataset = tf.data.Dataset, PaddedBatchDataset
            anchors = (total_anchors, [y1, x1, y2, x2])
                these values in normalized format between [0, 1]
            hyper_params = dictionary

        outputs:
            yield inputs, outputs
        """
    while True:
        for image_data in dataset:
            if cfg.MASK_REG:
                img, img_shape, gt_boxes, gt_labels, gt_mask, gt_seg_mask_inds = image_data
            else:
                img, img_shape, gt_boxes, gt_labels = image_data

            img_shape = tf.constant([[img.shape[1], img.shape[2]]])

            if cfg.MASK_REG:
                yield (img, img_shape, gt_boxes, gt_labels, gt_mask, gt_seg_mask_inds)
            else:
                yield (img, img_shape, gt_boxes, gt_labels)


def iit_generator_no_resize(dataset, cfg, base_anchors):
    """Tensorflow data generator for fit method, yielding inputs and outputs.
        inputs:
            dataset = tf.data.Dataset, PaddedBatchDataset
            anchors = (total_anchors, [y1, x1, y2, x2])
                these values in normalized format between [0, 1]
            hyper_params = dictionary

        outputs:
            yield inputs, outputs
        """
    while True:
        for image_data in dataset:
            if cfg.MASK_REG:
                img, img_shape, gt_boxes, gt_labels, gt_mask, gt_seg_mask_inds = image_data
            else:
                img, img_shape, gt_boxes, gt_labels = image_data

            img_shape = tf.constant([[img.shape[1], img.shape[2]]])
            feature_map_shape = (tf.cast(tf.floor(img.shape[1]/16), tf.int32), tf.cast(tf.floor(img.shape[2]/16), tf.int32))

            anchors = bbox_utils.generate_anchors(feature_map_shape, base_anchors)

            bbox_deltas, bbox_labels = calculate_rpn_actual_outputs_no_resize(anchors, gt_boxes, gt_labels, cfg, feature_map_shape)

            if cfg.MASK_REG:
                yield (img, img_shape, gt_boxes, gt_labels, bbox_deltas, bbox_labels, gt_mask, gt_seg_mask_inds), \
                      (tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32),
                       tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32),
                       tf.constant(0., dtype=tf.float32))
            else:
                yield (img, img_shape, gt_boxes, gt_labels, bbox_deltas, bbox_labels), \
                      (tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32),
                       tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32))


def iit_generator_no_resize_extra_loss(dataset, cfg, base_anchors):
    """Tensorflow data generator for fit method, yielding inputs and outputs.
        inputs:
            dataset = tf.data.Dataset, PaddedBatchDataset
            anchors = (total_anchors, [y1, x1, y2, x2])
                these values in normalized format between [0, 1]
            hyper_params = dictionary

        outputs:
            yield inputs, outputs
        """
    while True:
        for image_data in dataset:
            if cfg.MASK_REG:
                img, img_shape, gt_boxes, gt_labels, gt_mask, gt_seg_mask_inds = image_data
            else:
                img, img_shape, gt_boxes, gt_labels = image_data

            img_shape = tf.constant([[img.shape[1], img.shape[2]]])
            feature_map_shape = (tf.cast(tf.floor(img.shape[1]/16), tf.int32), tf.cast(tf.floor(img.shape[2]/16), tf.int32))

            anchors = bbox_utils.generate_anchors(feature_map_shape, base_anchors)

            bbox_deltas, bbox_labels = calculate_rpn_actual_outputs_no_resize(anchors, gt_boxes, gt_labels, cfg, feature_map_shape)

            if cfg.MASK_REG:
                yield (img, img_shape, gt_boxes, gt_labels, bbox_deltas, bbox_labels, gt_mask, gt_seg_mask_inds), \
                      (tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32),
                       tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32),
                       tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32))
            else:
                yield (img, img_shape, gt_boxes, gt_labels, bbox_deltas, bbox_labels), \
                      (tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32),
                       tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32))


def faster_rcnn_generator(dataset, anchors, cfg):
    """Tensorflow data generator for fit method, yielding inputs and outputs.
    inputs:
        dataset = tf.data.Dataset, PaddedBatchDataset
        anchors = (total_anchors, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        hyper_params = dictionary

    outputs:
        yield inputs, outputs
    """
    while True:
        for image_data in dataset:
            img, gt_boxes, gt_labels = image_data
            # print('image_data', image_data)
            bbox_deltas, bbox_labels = calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, cfg)
            yield (img, gt_boxes, gt_labels, bbox_deltas, bbox_labels), (tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32))

def rpn_generator(dataset, anchors, hyper_params):
    """Tensorflow data generator for fit method, yielding inputs and outputs.
    inputs:
        dataset = tf.data.Dataset, PaddedBatchDataset
        anchors = (total_anchors, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        hyper_params = dictionary

    outputs:
        yield inputs, outputs
    """
    while True:
        for image_data in dataset:
            img, gt_boxes, gt_labels = image_data
            bbox_deltas, bbox_labels = calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params)
            yield img, (bbox_deltas, bbox_labels)

def calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, cfg):
    """Generating one step data for training or inference.
    Batch operations supported.
    inputs:
        anchors = (total_anchors, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        gt_boxes (batch_size, gt_box_size, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        gt_labels (batch_size, gt_box_size)
        hyper_params = dictionary

    outputs:
        bbox_deltas = (batch_size, total_anchors, [delta_y, delta_x, delta_h, delta_w])
        bbox_labels = (batch_size, feature_map_shape, feature_map_shape, anchor_count)
    """
    batch_size = tf.shape(gt_boxes)[0]
    feature_map_shape = cfg.FEATURE_MAP_SHAPE
    anchor_count = cfg.ANCHOR_COUNT
    total_pos_bboxes = int(cfg.RPN_BATCHSIZE * cfg.RPN_FG_FRACTION)
    total_neg_bboxes = cfg.RPN_BATCHSIZE - total_pos_bboxes
    variances = cfg.VARIANCES
    # Calculate iou values between each bboxes and ground truth boxes
    iou_map, _ = bbox_utils.generate_iou_map(anchors, gt_boxes)
    # Get max index value for each row
    max_indices_each_row = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # Get max index value for each column
    max_indices_each_column = tf.argmax(iou_map, axis=1, output_type=tf.int32)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    #
    pos_mask = tf.greater(merged_iou_map, cfg.RPN_POSITIVE_OVERLAP)
    #
    valid_indices_cond = tf.not_equal(gt_labels, -1)
    valid_indices = tf.cast(tf.where(valid_indices_cond), tf.int32)
    valid_max_indices = max_indices_each_column[valid_indices_cond]
    #
    scatter_bbox_indices = tf.stack([valid_indices[..., 0], valid_max_indices], 1)
    max_pos_mask = tf.scatter_nd(scatter_bbox_indices, tf.fill((tf.shape(valid_indices)[0], ), True), tf.shape(pos_mask))
    pos_mask = tf.logical_or(pos_mask, max_pos_mask)
    pos_mask = randomly_select_xyz_mask(pos_mask, tf.constant([total_pos_bboxes], dtype=tf.int32))
    #
    pos_count = tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=-1)
    neg_count = (total_pos_bboxes + total_neg_bboxes) - pos_count
    #
    neg_mask = tf.logical_and(tf.less(merged_iou_map, cfg.RPN_NEGATIVE_OVERLAP), tf.logical_not(pos_mask))
    neg_mask = randomly_select_xyz_mask(neg_mask, neg_count)
    #
    pos_labels = tf.where(pos_mask, tf.ones_like(pos_mask, dtype=tf.float32), tf.constant(-1.0, dtype=tf.float32))
    neg_labels = tf.cast(neg_mask, dtype=tf.float32)
    bbox_labels = tf.add(pos_labels, neg_labels)
    #
    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_row, batch_dims=1)
    # Replace negative bboxes with zeros
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    # Calculate delta values between anchors and ground truth bboxes
    bbox_deltas = bbox_utils.get_deltas_from_bboxes(anchors, expanded_gt_boxes) / variances
    #
    # bbox_deltas = tf.reshape(bbox_deltas, (batch_size, feature_map_shape, feature_map_shape, anchor_count * 4))
    bbox_labels = tf.reshape(bbox_labels, (batch_size, feature_map_shape, feature_map_shape, anchor_count))
    #
    return bbox_deltas, bbox_labels

def calculate_rpn_actual_outputs_no_resize(anchors, gt_boxes, gt_labels, cfg, feature_map_shape):
    """Generating one step data for training or inference.
    Batch operations supported.
    inputs:
        anchors = (total_anchors, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        gt_boxes (batch_size, gt_box_size, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        gt_labels (batch_size, gt_box_size)
        hyper_params = dictionary

    outputs:
        bbox_deltas = (batch_size, total_anchors, [delta_y, delta_x, delta_h, delta_w])
        bbox_labels = (batch_size, feature_map_shape, feature_map_shape, anchor_count)
    """
    batch_size = tf.shape(gt_boxes)[0]
    anchor_count = cfg.ANCHOR_COUNT
    total_pos_bboxes = int(cfg.RPN_BATCHSIZE * cfg.RPN_FG_FRACTION)
    total_neg_bboxes = cfg.RPN_BATCHSIZE - total_pos_bboxes
    variances = cfg.VARIANCES
    # Calculate iou values between each bboxes and ground truth boxes
    iou_map, _ = bbox_utils.generate_iou_map(anchors, gt_boxes)
    # Get max index value for each row
    max_indices_each_row = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # Get max index value for each column
    max_indices_each_column = tf.argmax(iou_map, axis=1, output_type=tf.int32)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    #
    pos_mask = tf.greater(merged_iou_map, cfg.RPN_POSITIVE_OVERLAP)
    #
    valid_indices_cond = tf.not_equal(gt_labels, -1)
    valid_indices = tf.cast(tf.where(valid_indices_cond), tf.int32)
    valid_max_indices = max_indices_each_column[valid_indices_cond]
    #
    scatter_bbox_indices = tf.stack([valid_indices[..., 0], valid_max_indices], 1)
    max_pos_mask = tf.scatter_nd(scatter_bbox_indices, tf.fill((tf.shape(valid_indices)[0], ), True), tf.shape(pos_mask))
    pos_mask = tf.logical_or(pos_mask, max_pos_mask)
    pos_mask = randomly_select_xyz_mask(pos_mask, tf.constant([total_pos_bboxes], dtype=tf.int32))
    #
    pos_count = tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=-1)
    neg_count = (total_pos_bboxes + total_neg_bboxes) - pos_count
    #
    neg_mask = tf.logical_and(tf.less(merged_iou_map, cfg.RPN_NEGATIVE_OVERLAP), tf.logical_not(pos_mask))
    neg_mask = randomly_select_xyz_mask(neg_mask, neg_count)
    #
    pos_labels = tf.where(pos_mask, tf.ones_like(pos_mask, dtype=tf.float32), tf.constant(-1.0, dtype=tf.float32))
    neg_labels = tf.cast(neg_mask, dtype=tf.float32)
    bbox_labels = tf.add(pos_labels, neg_labels)
    #
    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_row, batch_dims=1)
    # Replace negative bboxes with zeros
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    # Calculate delta values between anchors and ground truth bboxes
    bbox_deltas = bbox_utils.get_deltas_from_bboxes(anchors, expanded_gt_boxes) / variances
    #
    # bbox_deltas = tf.reshape(bbox_deltas, (batch_size, feature_map_shape, feature_map_shape, anchor_count * 4))
    bbox_labels = tf.reshape(bbox_labels, (batch_size, feature_map_shape[0], feature_map_shape[1], anchor_count))
    #
    return bbox_deltas, bbox_labels


def rpn_cls_loss(*args):
    """Calculating rpn class loss value.
    Rpn actual class value should be 0 or 1.
    Because of this we only take into account non -1 values.
    inputs:
        *args = could be (y_true, y_pred) or ((y_true, y_pred), )

    outputs:
        loss = BinaryCrossentropy value
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    # remove -1 values
    indices = tf.where(tf.not_equal(y_true, tf.constant(-1.0, dtype=tf.float32)))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    lf = keras.losses.BinaryCrossentropy()
    return lf(target, output)


def rpn_reg_loss(*args):
    """Calculating rpn / faster rcnn regression loss value.
    Reg value should be different than zero for actual values.
    Because of this we only take into account non zero values.
    inputs:
        *args = could be (y_true, y_pred) or ((y_true, y_pred), )
        y_true (batch_size, total_anchors, deltas) = (2, 8649, 4)
        y_pred (batch_size, fm, fm, deltas*anchor_count) = (2, 31, 31, 36)
        target_labels  (batch_size, fm, fm, anchor_count) = (2, 31, 31, 9)
    outputs:
        loss = smooth L1 loss
    """
    y_true, y_pred, target_labels = args if len(args) == 3 else args[0]

    # Reshape to
    target_labels = tf.reshape(target_labels, (tf.shape(target_labels)[0], -1))  # (batch_size, num_rois)
    y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], tf.shape(y_true)[1], 4))  # (batch_size, num_rois, 4)

    # Reshape to merge batch and roi dimensions for simplicity.
    y_pred = tf.reshape(y_pred, (-1, 4))
    y_true = tf.reshape(y_true, (-1, 4))
    target_labels = tf.reshape(target_labels, (-1,))

    # Only positive ROIs contribute to the loss.
    positive_roi_ix = tf.where(target_labels > 0)[:, 0]

    # Gather the deltas (predicted and true) that contribute to loss
    y_true = tf.gather(y_true, positive_roi_ix)
    y_pred = tf.gather(y_pred, positive_roi_ix)

    loss = keras.backend.switch(tf.size(y_true) > tf.constant(0),
                                smooth_l1_loss(y_true=y_true, y_pred=y_pred),
                                tf.constant(0.0))
    loss = keras.backend.mean(loss)
    return loss


def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_pred - y_true)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * tf.abs(diff - 0.5)
    return loss


def cls_loss(*args):
    """Calculating faster rcnn class loss value.
    inputs:
        *args = could be (y_true, y_pred) or ((y_true, y_pred), )
        y_true (batch_size, num_rois)
        y_pred (batch_size, num_rois, 2)
    outputs:
        loss = CategoricalCrossentropy value
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    lf = keras.losses.CategoricalCrossentropy()
    return lf(y_true, y_pred)


def reg_loss(*args):
    """Calculating rpn / faster rcnn regression loss value.
    Reg value should be different than zero for actual values.
    Because of this we only take into account non zero values.
    inputs:
        *args = could be (y_true, y_pred) or ((y_true, y_pred), )
        y_true (batch_size, num_rois, deltas) = (2, 1500, 4)
        y_pred (batch_size, num_rois, deltas*num_classes) = (2, 1500, 84)
        target_labels  (batch_size, num_rois, num_classes) = (2, 1500, 21)
    outputs:
        loss = smooth L1 loss
    """
    y_true, y_pred, target_labels = args if len(args) == 3 else args[0]

    # Reshape to (batch_size, num_rois, num_classes, 4)
    y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], tf.shape(y_true)[1], -1, 4))

    # Get target labels -> decode from one hot vector
    target_labels = tf.argmax(target_labels, axis=2)    # (batch_size, num_rois)

    # Reshape to merge batch and roi dimensions for simplicity.
    y_pred = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], 4))
    y_true = tf.reshape(y_true, (-1, 4))
    target_labels = tf.reshape(target_labels, (-1,))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_labels > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_labels, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    y_true = tf.gather(y_true, positive_roi_ix)
    y_pred = tf.gather_nd(y_pred, indices)

    loss = keras.backend.switch(tf.size(y_true) > tf.constant(0),
                                smooth_l1_loss(y_true=y_true, y_pred=y_pred),
                                tf.constant(0.0))
    loss = keras.backend.mean(loss)
    return loss


def affordance_mask_loss(*args):
    """Loss for AffordanceNet mask.
    inputs:
        target_mask (batch_size, num_positive_rois, 244, 244)
        pred_mask_prob (batch_size, num_positive_rois, 224, 224, 11)
    """
    target_mask, pred_mask_prob = args if len(args) == 2 else args[0]
    target_mask = tf.reshape(target_mask, [tf.shape(target_mask)[0], tf.shape(pred_mask_prob)[1], -1])
    pred_mask_prob = tf.reshape(pred_mask_prob, [tf.shape(target_mask)[0], tf.shape(pred_mask_prob)[1], -1,
                                                 tf.shape(pred_mask_prob)[4]])
    loss = KLoss.SparseCategoricalCrossentropy()
    return loss(target_mask, pred_mask_prob)


def affordance_context_attr_loss(*args):
    """Calculating attribute class loss value.
    Attribute actual class value should be between 0 and 9 but in one hot representation.
    inputs:
        *args = could be (y_true, y_pred) or ((y_true, y_pred), )

    outputs:
        loss = CrossEntropy value
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    lf = keras.losses.BinaryCrossentropy()
    return lf(y_true, y_pred)


def affordance_context_reg_loss(*args):
    """Calculating rpn / faster rcnn regression loss value.
    Reg value should be different than zero for actual values.
    Because of this we only take into account non zero values.
    inputs:
        *args = could be (y_true, y_pred) or ((y_true, y_pred), )
        y_true (batch_size, num_rois, deltas) = (2, 1500, 4)
        y_pred (batch_size, num_rois, deltas*num_classes) = (2, 1500, 84)
        target_labels  (batch_size, num_rois, num_classes) = (2, 1500, 21)
    outputs:
        loss = smooth L1 loss
    """
    y_true, y_pred, target_labels = args if len(args) == 3 else args[0]

    # Reshape to (batch_size, num_rois, num_classes, 4)
    y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], tf.shape(y_true)[1], -1, 4))

    # Reshape to merge batch and roi dimensions for simplicity.
    y_pred = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], 4))
    y_true = tf.reshape(y_true, (-1, 4))
    target_labels = tf.reshape(target_labels, (-1,))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_labels > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_labels, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    y_true = tf.gather(y_true, positive_roi_ix)
    y_pred = tf.gather_nd(y_pred, indices)

    loss = keras.backend.switch(tf.size(y_true) > tf.constant(0),
                                smooth_l1_loss(y_true=y_true, y_pred=y_pred),
                                tf.constant(0.0))
    loss = keras.backend.mean(loss)
    return loss
