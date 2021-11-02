import tensorflow as tf


def generate_base_anchors(cfg):
    """
        Generating top left anchors for given anchor_ratios, anchor_scales and image size values.
    :param cfg: dictionary with configuration parameters
    :returns: base_anchors = (anchor_count, [y1, x1, y2, x2])
    """
    img_size = cfg.IMG_SIZE_WIDTH
    anchor_ratios = cfg.ANCHOR_RATIOS
    anchor_scales = cfg.ANCHOR_SCALES
    base_anchors = []
    for scale in anchor_scales:
        scale /= img_size
        for ratio in anchor_ratios:
            w = tf.sqrt(scale ** 2 / ratio)
            h = w * ratio
            base_anchors.append([-h / 2, -w / 2, h / 2, w / 2])
    return tf.cast(base_anchors, dtype=tf.float32)


def generate_anchors(feature_map_shape, base_anchors):
    """
        Broadcasting base_anchors and generating all anchors for given image parameters.
    :param feature_map_shape: shape of feature maps
    :param base_anchors: base anchors previously generated
    :returns: anchors = (output_width * output_height * anchor_count, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
    """
    feature_map_shape_x, feature_map_shape_y = feature_map_shape[0], feature_map_shape[1]
    stride_x = 1 / feature_map_shape_x
    stride_y = 1 / feature_map_shape_y
    grid_coords_x = tf.cast(tf.range(0, feature_map_shape_x) / feature_map_shape_x + stride_x / 2, dtype=tf.float32)
    grid_coords_y = tf.cast(tf.range(0, feature_map_shape_y) / feature_map_shape_y + stride_y / 2, dtype=tf.float32)

    grid_x, grid_y = tf.meshgrid(grid_coords_x, grid_coords_y)
    flat_grid_x, flat_grid_y = tf.reshape(grid_x, (-1, )), tf.reshape(grid_y, (-1, ))
    grid_map = tf.stack([flat_grid_y, flat_grid_x, flat_grid_y, flat_grid_x], axis=-1)

    anchors = tf.reshape(base_anchors, (1, -1, 4)) + tf.reshape(grid_map, (-1, 1, 4))
    anchors = tf.reshape(anchors, (-1, 4))
    return tf.clip_by_value(anchors, 0, 1)


def non_max_suppression(pred_bboxes, pred_labels, **kwargs):
    """Applying non maximum suppression.
    Details could be found on tensorflow documentation.
    https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression
    inputs:
        pred_bboxes = (batch_size, total_bboxes, total_labels, [y1, x1, y2, x2])
            total_labels should be 1 for binary operations like in rpn
        pred_labels = (batch_size, total_bboxes, total_labels)
        **kwargs = other parameters

    outputs:
        nms_boxes = (batch_size, max_detections, [y1, x1, y2, x2])
        nmsed_scores = (batch_size, max_detections)
        nmsed_classes = (batch_size, max_detections)
        valid_detections = (batch_size)
            Only the top valid_detections[i] entries in nms_boxes[i], nms_scores[i] and nms_class[i] are valid.
            The rest of the entries are zero paddings.
    """
    return tf.image.combined_non_max_suppression(
        pred_bboxes,
        pred_labels,
        **kwargs
    )


def get_bboxes_from_deltas(anchors, deltas):
    """Calculating bounding boxes for given bounding box and delta values.
    inputs:
        anchors = (batch_size, total_bboxes, [y1, x1, y2, x2])
        deltas = (batch_size, total_bboxes, [delta_y, delta_x, delta_h, delta_w])

    outputs:
        final_boxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
    """
    all_anc_width = anchors[..., 3] - anchors[..., 1]
    all_anc_height = anchors[..., 2] - anchors[..., 0]
    all_anc_ctr_x = anchors[..., 1] + 0.5 * all_anc_width
    all_anc_ctr_y = anchors[..., 0] + 0.5 * all_anc_height

    all_bbox_width = tf.exp(deltas[..., 3]) * all_anc_width
    all_bbox_height = tf.exp(deltas[..., 2]) * all_anc_height
    all_bbox_ctr_x = (deltas[..., 1] * all_anc_width) + all_anc_ctr_x
    all_bbox_ctr_y = (deltas[..., 0] * all_anc_height) + all_anc_ctr_y

    y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
    x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
    y2 = all_bbox_height + y1
    x2 = all_bbox_width + x1

    return tf.stack([y1, x1, y2, x2], axis=-1)


def get_deltas_from_bboxes(bboxes, gt_boxes):
    """Calculating bounding box deltas for given bounding box and ground truth boxes.
    inputs:
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
        gt_boxes = (batch_size, total_bboxes, [y1, x1, y2, x2])

    outputs:
        final_deltas = (batch_size, total_bboxes, [delta_y, delta_x, delta_h, delta_w])
    """
    bbox_width = bboxes[..., 3] - bboxes[..., 1]
    bbox_height = bboxes[..., 2] - bboxes[..., 0]
    bbox_ctr_x = bboxes[..., 1] + 0.5 * bbox_width
    bbox_ctr_y = bboxes[..., 0] + 0.5 * bbox_height

    gt_width = gt_boxes[..., 3] - gt_boxes[..., 1]
    gt_height = gt_boxes[..., 2] - gt_boxes[..., 0]
    gt_ctr_x = gt_boxes[..., 1] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[..., 0] + 0.5 * gt_height

    bbox_width = tf.where(tf.equal(bbox_width, 0), 1e-3, bbox_width)
    bbox_height = tf.where(tf.equal(bbox_height, 0), 1e-3, bbox_height)
    delta_x = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.truediv((gt_ctr_x - bbox_ctr_x), bbox_width))
    delta_y = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.truediv((gt_ctr_y - bbox_ctr_y), bbox_height))
    delta_w = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.math.log(gt_width / bbox_width))
    delta_h = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.math.log(gt_height / bbox_height))

    return tf.stack([delta_y, delta_x, delta_h, delta_w], axis=-1)


def generate_iou_map(bboxes, gt_boxes):
    """Calculating iou values for each ground truth boxes in batched manner.
    inputs:
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
        gt_boxes = (batch_size, total_gt_boxes, [y1, x1, y2, x2])

    outputs:
        iou_map = (batch_size, total_bboxes, total_gt_boxes)
        zero_iou = True if all union areas are 0, False otherwise
    """
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(bboxes, 4, axis=-1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1)
    # Calculate bbox and ground truth boxes areas
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis=-1)
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1)

    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, [0, 2, 1]))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))
    # Calculate intersection area
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    # Calculate union area
    union_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, 1) - intersection_area)
    # Intersection over Union
    if tf.equal(tf.reduce_sum(union_area), 0):
        return intersection_area / union_area, True
    return intersection_area / union_area, False


def normalize_bboxes(bboxes, height, width):
    """Normalizing bounding boxes.
       Maximum normalize coordinate is 1.0 in order to avoid possible errors in data (coordinate > height or width)
    inputs:
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
        height = image height
        width = image width
    outputs:
        normalized_bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
            in normalized form [0, 1]
    """
    y1 = bboxes[..., 0] / height
    x1 = bboxes[..., 1] / width
    y2 = bboxes[..., 2] / height
    x2 = bboxes[..., 3] / width
    return tf.stack([y1, x1, y2, x2], axis=-1)


def denormalize_bboxes(bboxes, height, width):
    """Denormalizing bounding boxes.
       Maximum denormalize coordinate is height or width in order to avoid possible errors in data (norm coordinate>1.0)
    inputs:
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2]) in normalized form [0, 1]
        height = image height
        width = image width
    outputs:
        denormalized_bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
    """
    y1 = bboxes[..., 0] * height
    x1 = bboxes[..., 1] * width
    y2 = bboxes[..., 2] * height
    x2 = bboxes[..., 3] * width
    return tf.round(tf.stack([y1, x1, y2, x2], axis=-1))


def clip_bboxes(bboxes, height, width):
    """
        Clip boxes to image boundaries.
    :param bboxes: bboxes not normalized (batch_size, total_bboxes, [y1, x1, y2, x2])
    :param height: image height
    :param width: image width
    :returns: clipped_bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
    """
    # y1 >= 0, x1 >= 0
    y1 = tf.maximum(tf.minimum(bboxes[..., 0], height), 0)
    x1 = tf.maximum(tf.minimum(bboxes[..., 1], width), 0)
    # y2 < height, x2 < width
    y2 = tf.maximum(tf.minimum(bboxes[..., 2], height), 0)
    x2 = tf.maximum(tf.minimum(bboxes[..., 3], width), 0)
    return tf.stack([y1, x1, y2, x2], axis=-1)
