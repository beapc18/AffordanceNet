import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Lambda, Input, Conv2D, TimeDistributed, Dense, Flatten
from utils import bbox_utils, train_utils
import tensorflow.keras.regularizers as KR
import tensorflow.keras.layers as KL
import tensorflow.keras.initializers as KI


class Decoder(Layer):
    """Generating bounding boxes and labels from faster rcnn predictions.
    First calculating the boxes from predicted deltas and label probs.
    Then applied non max suppression and selecting top_n boxes by scores.
    In case all scores are lower than score_threshold -> take bbox with maximun score
    inputs:
        roi_bboxes = (batch_size, roi_bbox_size, [y1, x1, y2, x2])
        pred_deltas = (batch_size, roi_bbox_size, total_labels * [delta_y, delta_x, delta_h, delta_w])
        pred_label_probs = (batch_size, roi_bbox_size, total_labels)
    outputs:
        pred_bboxes = (batch_size, top_n, [y1, x1, y2, x2])
        pred_labels = (batch_size, top_n)
            1 to total label number
        pred_scores = (batch_size, top_n)
    """
    def __init__(self, variances, total_labels, nms_threshold=0.5, max_total_size=200, score_threshold=0.5, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.variances = variances
        self.total_labels = total_labels
        self.nms_threshold = nms_threshold
        self.max_total_size = max_total_size
        self.score_threshold = score_threshold

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            "variances": self.variances,
            "total_labels": self.total_labels,
            "nms_threshold": self.nms_threshold,
            "max_total_size": self.max_total_size,
            "score_threshold": self.score_threshold
        })
        return config

    def call(self, inputs):
        roi_bboxes = inputs[0]
        pred_deltas = inputs[1]
        pred_label_probs = inputs[2]
        batch_size = tf.shape(pred_deltas)[0]

        pred_deltas = tf.reshape(pred_deltas, (batch_size, -1, self.total_labels, 4))
        pred_deltas *= self.variances

        expanded_roi_bboxes = tf.tile(tf.expand_dims(roi_bboxes, -2), (1, 1, self.total_labels, 1))
        pred_bboxes = bbox_utils.get_bboxes_from_deltas(expanded_roi_bboxes, pred_deltas)

        pred_labels_map = tf.expand_dims(tf.argmax(pred_label_probs, -1), -1)
        pred_labels = tf.where(tf.not_equal(pred_labels_map, 0), pred_label_probs, tf.zeros_like(pred_label_probs))

        final_bboxes, final_scores, final_labels, valid_detections = bbox_utils.non_max_suppression(
                                                                    pred_bboxes, pred_labels,
                                                                    iou_threshold=self.nms_threshold,
                                                                    max_output_size_per_class=self.max_total_size,
                                                                    max_total_size=self.max_total_size,
                                                                    score_threshold=self.score_threshold)

        # If there are any valid detection -> Apply NMS but without score threshold
        no_detections = valid_detections[0] == 0
        if no_detections:
            final_bboxes, final_scores, final_labels, valid_detections = bbox_utils.non_max_suppression(pred_bboxes,
                                                                       pred_labels, iou_threshold=self.nms_threshold,
                                                                       max_output_size_per_class=self.max_total_size,
                                                                       max_total_size=self.max_total_size)

        # Take only valid outputs, remove zero padding -> only valid for batchsize=1
        if batch_size == 1:
            final_bboxes = tf.slice(final_bboxes, [0, 0, 0], [1, valid_detections[0], 4])
            final_scores = tf.slice(final_scores, [0, 0], [1, valid_detections[0]])
            final_labels = tf.slice(final_labels, [0, 0], [1, valid_detections[0]])

        if no_detections:
            best_score = tf.reduce_max(final_scores, axis=1)
            if best_score < 0.001:  # no good bbox
                final_bboxes = tf.zeros((1, 1, 4))
                final_labels = tf.zeros((1, 1))
                final_scores = tf.zeros((1, 1))
            else:
                better_detection_index = tf.argmax(final_scores, axis=1)
                final_bboxes = tf.gather(final_bboxes, better_detection_index, axis=1)
                final_scores = tf.gather(final_scores, better_detection_index, axis=1)
                final_labels = tf.gather(final_labels, better_detection_index, axis=1)

        return final_bboxes, final_labels, final_scores


class ProposalLayer(Layer):
    """Generating bounding boxes from rpn predictions.
    First calculating the boxes from predicted deltas and label probs.
    Then applied non max suppression and selecting "train or test nms_topn" boxes.
    inputs:
        rpn_bbox_deltas = (batch_size, img_output_height, img_output_width, anchor_count * [delta_y, delta_x, delta_h, delta_w])
            img_output_height and img_output_width are calculated to the base model feature map
        rpn_labels = (batch_size, img_output_height, img_output_width, anchor_count)

    outputs:
        roi_bboxes = (batch_size, train/test_nms_topn, [y1, x1, y2, x2])
    """

    def __init__(self, base_anchors, mode, cfg, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.base_anchors = base_anchors
        self.cfg = cfg
        self.mode = mode

    def get_config(self):
        config = super(ProposalLayer, self).get_config()
        config.update({"base_anchors": self.base_anchors, "cfg": self.cfg, "mode": self.mode})
        return config

    def call(self, inputs):
        rpn_bbox_deltas = inputs[0]
        rpn_labels = inputs[1]
        anchors = bbox_utils.generate_anchors((tf.shape(rpn_labels)[1], tf.shape(rpn_labels)[2]), self.base_anchors)

        pre_nms_topn = self.cfg.PRE_NMS_TOPN if self.mode == "training" else self.cfg.TEST_PRE_NMS_TOPN
        post_nms_topn = self.cfg.TRAIN_NMS_TOPN if self.mode == "training" else self.cfg.TEST_NMS_TOPN
        nms_iou_threshold = self.cfg.NMS_IOU_THRESHOLD
        variances = self.cfg.VARIANCES
        total_anchors = tf.shape(anchors)[0]
        batch_size = tf.shape(rpn_bbox_deltas)[0]
        rpn_bbox_deltas = tf.reshape(rpn_bbox_deltas, (batch_size, total_anchors, 4))
        rpn_labels = tf.reshape(rpn_labels, (batch_size, total_anchors))

        rpn_bbox_deltas *= variances
        rpn_bboxes = bbox_utils.get_bboxes_from_deltas(anchors, rpn_bbox_deltas)

        # if there are less possible anchors than pre nms, then take all of them
        if tf.shape(rpn_labels)[1] < pre_nms_topn:
            pre_nms_topn = tf.shape(rpn_labels)[1]

        _, pre_indices = tf.nn.top_k(rpn_labels, pre_nms_topn)

        # take top rois and apply NMS
        pre_roi_bboxes = tf.gather(rpn_bboxes, pre_indices, batch_dims=1)
        pre_roi_labels = tf.gather(rpn_labels, pre_indices, batch_dims=1)

        pre_roi_bboxes = tf.reshape(pre_roi_bboxes, (batch_size, pre_nms_topn, 1, 4))
        pre_roi_labels = tf.reshape(pre_roi_labels, (batch_size, pre_nms_topn, 1))

        roi_bboxes, _, _, _ = bbox_utils.non_max_suppression(pre_roi_bboxes, pre_roi_labels,
                                                             max_output_size_per_class=post_nms_topn,
                                                             max_total_size=post_nms_topn,
                                                             iou_threshold=nms_iou_threshold)
        return tf.stop_gradient(roi_bboxes)


class ProposalTargetLayer(Layer):
    """Calculating faster rcnn actual bounding box deltas and labels.
    This layer only running on the training phase.
    inputs:
        roi_bboxes = (batch_size, nms_topn, [y1, x1, y2, x2])
        gt_boxes = (batch_size, padded_gt_boxes_size, [y1, x1, y2, x2])
        gt_labels = (batch_size, padded_gt_boxes_size)
        gt_masks = (batch_size, num_masks, img_height, img_width)

    outputs:
        roi_bbox_deltas = (batch_size, train_nms_topn * total_labels, [delta_y, delta_x, delta_h, delta_w])
        roi_bbox_labels = (batch_size, train_nms_topn, total_labels)
    """

    def __init__(self, cfg, img_height, img_width, **kwargs):
        super(ProposalTargetLayer, self).__init__(**kwargs)
        self.cfg = cfg
        self.img_height = img_height
        self.img_width = img_width

    def get_config(self):
        config = super(ProposalTargetLayer, self).get_config()
        config.update({"cfg": self.cfg, "img_height": self.img_height, "img_width": self.img_width})
        return config

    def call(self, inputs):
        img_shape = inputs[0]
        roi_bboxes = inputs[1]
        gt_boxes = inputs[2]
        gt_labels = inputs[3]
        if self.cfg.MASK_REG:
            gt_masks = inputs[4]

        total_labels = self.cfg.NUM_CLASSES
        variances = self.cfg.VARIANCES

        # Calculate iou values between each bboxes and ground truth boxes
        iou_map, _ = bbox_utils.generate_iou_map(roi_bboxes, gt_boxes)
        # Get max index value for each row
        max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
        # IoU map has iou values for every gt boxes and we merge these values column wise
        merged_iou_map = tf.reduce_max(iou_map, axis=2)

        # select positive and negative rois according to the thresholds
        pos_mask = tf.greater(merged_iou_map, self.cfg.TRAIN_FG_THRES)
        neg_mask = tf.logical_and(tf.less(merged_iou_map, self.cfg.TRAIN_BG_THRESH_HI), tf.greater(merged_iou_map, self.cfg.TRAIN_BG_THRESH_LO))

        # Calculate positive and negative total number of rois
        positive_count = tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=1)
        max_pos_bboxes = tf.cast(tf.round(self.cfg.TRAIN_ROIS_PER_IMAGE*self.cfg.ROI_POSITIVE_RATIO), tf.int32)
        total_pos_bboxes = tf.minimum(max_pos_bboxes, positive_count)
        negative_count = tf.reduce_sum(tf.cast(neg_mask, tf.int32), axis=1)
        negative_max2 = self.cfg.TRAIN_ROIS_PER_IMAGE - total_pos_bboxes
        total_neg_bboxes = tf.minimum(negative_max2, negative_count)
        positive_count = total_pos_bboxes[0]
        negative_count = total_neg_bboxes[0]

        # Take random positive and negative rois without replacement
        if positive_count > 0:
            pos_mask = train_utils.randomly_select_xyz_mask(pos_mask, total_pos_bboxes)
        if negative_count > 0:
            neg_mask = train_utils.randomly_select_xyz_mask(neg_mask, total_neg_bboxes)

        # take corresponding gt boxes and gt labels to rois
        gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
        expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, axis=-1), gt_boxes_map, tf.zeros_like(gt_boxes_map))

        gt_labels_map = tf.gather(gt_labels, max_indices_each_gt_box, batch_dims=1)
        pos_gt_labels = tf.where(pos_mask, gt_labels_map, tf.constant(-1, dtype=tf.int32))
        neg_gt_labels = tf.cast(neg_mask, dtype=tf.int32)
        expanded_gt_labels = tf.cast(pos_gt_labels + neg_gt_labels, dtype=tf.int32)     # (batch_size, num_rois, 4)

        # take positive gt bboxes, labels and rois
        pos_indices = tf.where(pos_mask)
        positive_count = tf.shape(pos_indices)[0]
        gt_boxes_pos = tf.gather_nd(expanded_gt_boxes, pos_indices)
        positive_rois = tf.gather_nd(roi_bboxes, pos_indices)
        pos_gt_labels = tf.gather_nd(expanded_gt_labels, pos_indices)

        # take negative gt bboxes, labels and rois
        neg_indices = tf.where(neg_mask)
        gt_boxes_neg = tf.gather_nd(expanded_gt_boxes, neg_indices)
        neg_rois = tf.gather_nd(roi_bboxes, neg_indices)
        neg_gt_labels = tf.gather_nd(expanded_gt_labels, neg_indices)

        # concat positive + negative gt bboxes, labels and rois
        total_gt_bboxes = tf.concat([gt_boxes_pos, gt_boxes_neg], 0)
        total_gt_labels = tf.concat([pos_gt_labels, neg_gt_labels], 0)
        total_rois = tf.concat([positive_rois, neg_rois], 0)

        # get deltas from bboxes
        gt_bbox_deltas = bbox_utils.get_deltas_from_bboxes(total_rois, total_gt_bboxes) / variances
        gt_bbox_labels = total_gt_labels

        # Transform to one hot representation (batch_size, num_rois, num_classes)
        gt_bbox_labels = tf.one_hot(gt_bbox_labels, total_labels)

        gt_bbox_deltas = tf.expand_dims(gt_bbox_deltas, axis=0)
        gt_bbox_labels = tf.expand_dims(gt_bbox_labels, axis=0)
        total_rois = tf.expand_dims(total_rois, axis=0)

        if self.cfg.MASK_REG:
            # Take only positive rois for mask training and corresponding roi_gt_boxes
            roi_gt_boxes = tf.gather_nd(gt_boxes_map, pos_indices)

            y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
            y1t, x1t, y2t, x2t = tf.split(roi_gt_boxes, 4, axis=1)

            # compute overlap between roi coordinate and gt_roi coordinate
            x1o = tf.maximum(x1, x1t)
            y1o = tf.maximum(y1, y1t)
            x2o = tf.minimum(x2, x2t)
            y2o = tf.minimum(y2, y2t)

            if positive_count != 0:
                # Calculate labels in original mask -> gt_masks=(batch_size, num_masks, img_height, img_width)
                original_affordance_labels = tf.unique(tf.reshape(gt_masks, [-1]))
                original_affordance_labels = tf.sort(original_affordance_labels.y)

                # filter indices of gt boxes
                indices_pos_gt_boxes = tf.boolean_mask(max_indices_each_gt_box, pos_mask)

                # mask associated wrt to gt bbox (batch_size, positive_rois, mask_size, mask_size)
                gt_mask = tf.gather(gt_masks, indices_pos_gt_boxes, axis=1)

                gt_mask = tf.cast(tf.expand_dims(gt_mask, axis=4), tf.float32)
                y1o = tf.squeeze(y1o, axis=1)
                x1o = tf.squeeze(x1o, axis=1)
                y2o = tf.squeeze(y2o, axis=1)
                x2o = tf.squeeze(x2o, axis=1)

                # create boxes to crop and indexes where each mask has its own box
                boxes = tf.cast(tf.stack([y1o, x1o, y2o, x2o], axis=1), tf.float32)

                # remove batch dim -> needed for crop and resize op
                img_shape = tf.squeeze(img_shape, axis=0)
                gt_mask = tf.squeeze(gt_mask, axis=0)

                # crop and resize the masks individually
                positive_masks = self._crop_and_resize_masks_no_resize(img_shape, gt_mask, boxes, positive_rois,
                                                                       positive_count, original_affordance_labels)
                # Add batch dim
                positive_masks = tf.expand_dims(positive_masks, axis=0)
                positive_rois = tf.expand_dims(positive_rois, axis=0)
                masks = positive_masks
            else:
                positive_rois = tf.expand_dims(positive_rois, axis=0)
                masks = tf.constant(0, dtype=tf.int32, shape=[1, 0, self.cfg.TRAIN_MASK_SIZE, self.cfg.TRAIN_MASK_SIZE])

            return total_rois, tf.stop_gradient(gt_bbox_deltas), tf.stop_gradient(gt_bbox_labels), tf.stop_gradient(masks), \
                tf.stop_gradient(positive_rois)

        return tf.stop_gradient(gt_bbox_deltas), tf.stop_gradient(gt_bbox_labels)

    def _crop_and_resize_masks_no_resize(self, img_shape, masks, overlapping_boxes, rois, positive_count, original_aff_labels):
        # denormalize bboxes
        overlapping_boxes = tf.cast(bbox_utils.denormalize_bboxes(overlapping_boxes, img_shape[0], img_shape[1]), tf.int32)
        rois = tf.cast(bbox_utils.denormalize_bboxes(rois, img_shape[0], img_shape[1]), tf.int32)

        num_masks = tf.shape(masks)[0]
        final_masks = tf.zeros((num_masks, self.cfg.TRAIN_MASK_SIZE, self.cfg.TRAIN_MASK_SIZE))
        for i in range(num_masks):
            mask = masks[i]

            # get roi and overlap area coordinates
            y1, x1, y2, x2 = tf.split(rois[i], 4, axis=0)
            y1, x1, y2, x2 = tf.squeeze(y1), tf.squeeze(x1), tf.squeeze(y2), tf.squeeze(x2)

            y1o, x1o, y2o, x2o = tf.split(overlapping_boxes[i], 4, axis=0)
            y1o, x1o, y2o, x2o = tf.squeeze(y1o), tf.squeeze(x1o), tf.squeeze(y2o), tf.squeeze(x2o)

            # take overlap area between gt_bbox and roi
            overlapping_mask_area = mask[y1o:y2o, x1o:x2o]

            # calculate offsets with 0 above and in the left of the overlapping area
            offset_height = y1o - y1
            offset_width = x1o - x1

            # calculate roi height and width
            target_height = y2 - y1 + 1
            target_width = x2 - x1 + 1

            roi_mask = tf.image.pad_to_bounding_box(overlapping_mask_area, offset_height, offset_width, target_height, target_width)

            # resize to mask size
            roi_mask = tf.image.resize(roi_mask, [self.cfg.TRAIN_MASK_SIZE, self.cfg.TRAIN_MASK_SIZE], method='bilinear')

            # Create a structure with 0 for all masks except for the current mask and add that to final mask structure
            temp_masks = tf.scatter_nd([[i]], tf.expand_dims(tf.squeeze(roi_mask, axis=2), axis=0),
                                       [num_masks, self.cfg.TRAIN_MASK_SIZE, self.cfg.TRAIN_MASK_SIZE])
            final_masks += temp_masks

        final_masks = self._convert_mask_to_original_ids_manual(positive_count, final_masks, original_aff_labels, self.cfg.TRAIN_MASK_SIZE)
        return final_masks

    def _reset_mask_ids(self, mask, before_uni_ids):
        # reset ID mask values from [0, 1, 4] to [0, 1, 2] to resize later
        counter = 0
        final_mask = tf.zeros_like(mask)
        for id in before_uni_ids:
            # mask[mask == id] = counter
            temp_mask = tf.where(mask == id, counter, 0)
            final_mask += temp_mask
            counter += 1
        return final_mask

    def _convert_mask_to_original_ids_manual(self, positive_count, mask, original_uni_ids, train_mask_size):
        const = 0.005
        original_uni_ids_2 = tf.expand_dims(tf.expand_dims(tf.expand_dims(original_uni_ids, axis=1), axis=2), axis=3)
        dif = tf.abs(mask - tf.cast(original_uni_ids_2, tf.float32)) < const
        max = tf.expand_dims(tf.argmax(dif, axis=0), axis=3)
        # create mask array where each position contains the original_uni_ids
        temp_mask = tf.where(tf.fill([positive_count, train_mask_size, train_mask_size, 1], 0) == 0, original_uni_ids, original_uni_ids)
        return tf.gather_nd(temp_mask, max, batch_dims=3)


class RoiAlign(Layer):
    """Reducing all feature maps to same size.
    Firstly cropping bounding boxes from the feature maps and then resizing it to the pooling size.
    inputs:
        feature_map = (batch_size, img_output_height, img_output_width, channels)
        roi_bboxes = (batch_size, train/test_nms_topn, [y1, x1, y2, x2])

    outputs:
        final_pooling_feature_map = (batch_size, train/test_nms_topn, pooling_size[0], pooling_size[1], channels)
            pooling_size usually (7, 7)
    """

    def __init__(self, cfg, **kwargs):
        super(RoiAlign, self).__init__(**kwargs)
        self.cfg = cfg

    def get_config(self):
        config = super(RoiAlign, self).get_config()
        config.update({"cfg": self.cfg})
        return config

    def call(self, inputs):
        feature_map = inputs[0]
        roi_bboxes = inputs[1]
        pooling_size = (self.cfg.POOL_SIZE, self.cfg.POOL_SIZE)
        batch_size, total_bboxes = tf.shape(roi_bboxes)[0], tf.shape(roi_bboxes)[1]
        row_size = batch_size * total_bboxes

        # We need to arange bbox indices for each batch
        pooling_bbox_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), (1, total_bboxes))
        pooling_bbox_indices = tf.reshape(pooling_bbox_indices, (-1,))
        pooling_bboxes = tf.reshape(roi_bboxes, (row_size, 4))

        # Crop to bounding box size then resize to pooling size
        # This method resize using bilinear interpolation with aligned corners
        pooling_feature_map = tf.image.crop_and_resize(
            feature_map,
            pooling_bboxes,
            pooling_bbox_indices,
            pooling_size
        )
        final_pooling_feature_map = tf.reshape(pooling_feature_map, (batch_size, total_bboxes,
                                               pooling_feature_map.shape[1], pooling_feature_map.shape[2],
                                               pooling_feature_map.shape[3]))
        return final_pooling_feature_map


def get_affordance_net_model(feature_extractor, rpn_model, cfg, base_anchors, mode="training"):
    """Generating rpn model for given backbone base model and hyper params.
    inputs:
        feature_extractor = feature extractor layer from the base model
        rpn_model = tf.keras.model generated rpn model
        anchors = (total_anchors, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        hyper_params = dictionary
        mode = "training" or "inference"

    outputs:
        frcnn_model = tf.keras.model
    """
    input_img = rpn_model.input
    rpn_reg_predictions, rpn_cls_predictions = rpn_model.output
    l2_reg = KR.l2(cfg.WEIGHT_DECAY)

    roi_bboxes = ProposalLayer(base_anchors, mode, cfg, name="roi_bboxes")([rpn_reg_predictions, rpn_cls_predictions])

    if mode == "training":
        input_img_shape = Input(shape=(2,), name="input_img_shape", dtype=tf.float32)
        input_gt_boxes = Input(shape=(None, 4), name="input_gt_boxes", dtype=tf.float32)
        input_gt_labels = Input(shape=(None, ), name="input_gt_labels", dtype=tf.int32)
        rpn_cls_actuals = Input(shape=(None, None, cfg.ANCHOR_COUNT), name="input_rpn_cls_actuals", dtype=tf.float32)
        rpn_reg_actuals = Input(shape=(None, 4), name="input_rpn_reg_actuals", dtype=tf.float32)
        input_gt_masks = KL.Input(shape=(None, cfg.IMG_SIZE_HEIGHT, cfg.IMG_SIZE_WIDTH), name="input_gt_masks", dtype=tf.int32)
        input_gt_seg_mask_inds = KL.Input(shape=(None,), name="input_gt_seg_mask_inds", dtype=tf.int32)

        rois, frcnn_reg_actuals, frcnn_cls_actuals, target_maks, rois_pos = ProposalTargetLayer(cfg, input_img_shape[0], input_img_shape[1], name="roi_deltas")(
            [input_img_shape, roi_bboxes, input_gt_boxes, input_gt_labels, input_gt_masks, input_gt_seg_mask_inds])

        roi_pooled = RoiAlign(cfg, name="roi_pooling")([feature_extractor.output, rois])

        output = TimeDistributed(Flatten(), name="frcnn_flatten")(roi_pooled)
        output = TimeDistributed(Dense(4096, activation="relu", kernel_regularizer=l2_reg), name="frcnn_fc1")(output)
        output = TimeDistributed(Dense(4096, activation="relu", kernel_regularizer=l2_reg), name="frcnn_fc2")(output)

        frcnn_cls_predictions = TimeDistributed(Dense(cfg.NUM_CLASSES, activation="softmax", kernel_regularizer=l2_reg,
                                                      kernel_initializer=KI.RandomNormal(stddev=0.01),
                                                      bias_initializer="zeros"), name="frcnn_cls")(output)
        frcnn_reg_predictions = TimeDistributed(Dense(cfg.NUM_CLASSES * 4, activation="linear", kernel_regularizer=l2_reg,
                                                      kernel_initializer=KI.RandomNormal(stddev=0.001),
                                                      bias_initializer="zeros"), name="frcnn_reg")(output)

        # only for positive rois
        roi_align_mask = RoiAlign(cfg, name="roi_align_mask")([feature_extractor.output, rois_pos])

    else:
        roi_pooled = RoiAlign(cfg, name="roi_pooling")([feature_extractor.output, roi_bboxes])

        output = TimeDistributed(Flatten(), name="frcnn_flatten")(roi_pooled)
        output = TimeDistributed(Dense(4096, activation="relu", kernel_regularizer=l2_reg), name="frcnn_fc1")(output)
        output = TimeDistributed(Dense(4096, activation="relu", kernel_regularizer=l2_reg), name="frcnn_fc2")(output)

        frcnn_cls_predictions = TimeDistributed(Dense(cfg.NUM_CLASSES, activation="softmax", kernel_regularizer=l2_reg),
                                                name="frcnn_cls")(output)
        frcnn_reg_predictions = TimeDistributed(
            Dense(cfg.NUM_CLASSES * 4, activation="linear", kernel_regularizer=l2_reg), name="frcnn_reg")(output)

        bboxes, labels, scores = Decoder(cfg.VARIANCES, cfg.NUM_CLASSES, nms_threshold=cfg.TEST_NMS_THRESHOLD,
                                         max_total_size=cfg.MAX_PER_IMAGE, score_threshold=cfg.SCORE_THRESHOLD,
                                         name="faster_rcnn_decoder")([roi_bboxes, frcnn_reg_predictions, frcnn_cls_predictions])
        roi_align_mask = RoiAlign(cfg, name="roi_align_mask")([feature_extractor.output, bboxes])

    pool5_2_conv = KL.TimeDistributed(KL.Conv2D(512, (1, 1), activation="relu", padding="valid",
                                                kernel_initializer=KI.RandomNormal(stddev=0.01),
                                                bias_initializer="zeros",
                                                kernel_regularizer=l2_reg),
                                      name="mask_conv_1")(roi_align_mask)
    pool5_2_conv2 = KL.TimeDistributed(KL.Conv2D(512, (3, 3), activation="relu", padding="same",
                                                 kernel_initializer=KI.RandomNormal(stddev=0.01),
                                                 bias_initializer="zeros",
                                                 kernel_regularizer=l2_reg),
                                       name="mask_conv_2")(pool5_2_conv)

    # how to calculate output dims for transpose conv
    # https://github.com/tensorflow/tensorflow/blob/5dcfc51118817f27fad5246812d83e5dccdc5f72/tensorflow/python/keras/utils/conv_utils.py#L139
    mask_deconv1 = KL.TimeDistributed(KL.Conv2DTranspose(256, (8, 8), padding="same", strides=(4, 4), groups=256, kernel_regularizer=l2_reg),
                                      name='mask_deconv_1')(pool5_2_conv2)

    pool5_2_conv3 = KL.TimeDistributed(KL.Conv2D(512, (3, 3), activation="relu", padding="same",
                                                 kernel_initializer=KI.RandomNormal(stddev=0.01),
                                                 bias_initializer="zeros",
                                                 kernel_regularizer=l2_reg),
                                       name="mask_conv_3")(mask_deconv1)

    pool5_2_conv4 = KL.TimeDistributed(KL.Conv2D(512, (3, 3), activation="relu", padding="same",
                                                 kernel_initializer=KI.RandomNormal(stddev=0.01),
                                                 bias_initializer="zeros",
                                                 kernel_regularizer=l2_reg),
                                       name="mask_conv_4")(pool5_2_conv3)

    mask_deconv2 = KL.TimeDistributed(KL.Conv2DTranspose(256, (8, 8), padding="same", strides=(4, 4), groups=256, kernel_regularizer=l2_reg),
                                                         # kernel_initializer=BilinearInitializer(filter_size=8, num_channels_in=512, num_channels_out=256)),
                                      name='mask_deconv_2')(pool5_2_conv4)

    pool5_2_conv5 = KL.TimeDistributed(KL.Conv2D(512, (3, 3), activation="relu", padding="same",
                                                 kernel_initializer=KI.RandomNormal(stddev=0.01),
                                                 bias_initializer="zeros",
                                                 kernel_regularizer=l2_reg),
                                       name="mask_conv_5")(mask_deconv2)

    pool5_2_conv6 = KL.TimeDistributed(KL.Conv2D(512, (3, 3), activation="relu", padding="same",
                                                 kernel_initializer=KI.RandomNormal(stddev=0.01),
                                                 bias_initializer="zeros",
                                                 kernel_regularizer=l2_reg),
                                       name="mask_conv_6")(pool5_2_conv5)

    mask_deconv3 = KL.TimeDistributed(KL.Conv2DTranspose(256, (4, 4), padding="same", strides=(2, 2), groups=256, kernel_regularizer=l2_reg),
                                                         # kernel_initializer=BilinearInitializer(filter_size=4, num_channels_in=512, num_channels_out=256)),
                                      name='mask_deconv_3')(pool5_2_conv6)

    mask_prob_output = KL.TimeDistributed(KL.Conv2D(cfg.NUM_AFFORDANCE_CLASSES, (1, 1), padding="valid", activation="softmax",
                                                    kernel_initializer=KI.RandomNormal(stddev=0.01),
                                                    bias_initializer="zeros",
                                                    kernel_regularizer=l2_reg),
                                          name="mask_score")(mask_deconv3)

    if mode == "training":
        rpn_cls_loss_layer = Lambda(train_utils.rpn_cls_loss, name=cfg.LOSS_NAMES[1])([rpn_cls_actuals, rpn_cls_predictions])
        rpn_reg_loss_layer = Lambda(train_utils.rpn_reg_loss, name=cfg.LOSS_NAMES[0])([rpn_reg_actuals, rpn_reg_predictions, rpn_cls_actuals])
        reg_loss_layer = Lambda(train_utils.reg_loss, name=cfg.LOSS_NAMES[2])([frcnn_reg_actuals, frcnn_reg_predictions, frcnn_cls_actuals])
        cls_loss_layer = Lambda(train_utils.cls_loss, name=cfg.LOSS_NAMES[3])([frcnn_cls_actuals, frcnn_cls_predictions])
        mask_loss_layer = KL.Lambda(train_utils.affordance_mask_loss, name=cfg.LOSS_NAMES[4])([target_maks, mask_prob_output])

        frcnn_model = CustomModel(cfg.LOSS_NAMES, cfg.LOSS_WEIGHTS,
                                  inputs=[input_img, input_img_shape,  input_gt_boxes, input_gt_labels, rpn_reg_actuals,
                                          rpn_cls_actuals, input_gt_masks, input_gt_seg_mask_inds],
                                  outputs=[rpn_reg_loss_layer, rpn_cls_loss_layer, reg_loss_layer,
                                           cls_loss_layer, mask_loss_layer])
    else:
        frcnn_model = Model(inputs=[input_img], outputs=[bboxes, labels, scores, mask_prob_output])
    return frcnn_model

def init_model(model, cfg):
    """Generating dummy data for initialize model.
    In this way, the training process can continue from where it left off.
    inputs:
        model = tf.keras.model
        hyper_params = dictionary
    """
    final_height, final_width = cfg.IMG_SIZE_HEIGHT, cfg.IMG_SIZE_WIDTH
    img = tf.random.uniform((cfg.BATCH_SIZE, final_height, final_width, 3))
    feature_map_shape = cfg.FEATURE_MAP_SHAPE
    total_anchors = feature_map_shape * feature_map_shape * cfg.ANCHOR_COUNT
    gt_boxes = tf.random.uniform((cfg.BATCH_SIZE, 1, 4))
    gt_labels = tf.random.uniform((cfg.BATCH_SIZE, 1), maxval=cfg.NUM_CLASSES, dtype=tf.int32)
    bbox_deltas = tf.random.uniform((cfg.BATCH_SIZE, total_anchors, 4))
    bbox_labels = tf.random.uniform((cfg.BATCH_SIZE, feature_map_shape, feature_map_shape, cfg.ANCHOR_COUNT), maxval=1, dtype=tf.float32)
    if cfg.MASK_REG:
        mask = tf.random.uniform((cfg.BATCH_SIZE, 1, final_height, final_width))
        mask_ids = tf.random.uniform((cfg.BATCH_SIZE, 1), maxval=cfg.NUM_AFFORDANCE_CLASSES, dtype=tf.int32)
        model([img, gt_boxes, gt_labels, bbox_deltas, bbox_labels, mask, mask_ids])
    else:
        model([img, gt_boxes, gt_labels, bbox_deltas, bbox_labels])

def init_model_no_resize(model, cfg):
    """Generating dummy data for initialize model.
    In this way, the training process can continue from where it left off.
    inputs:
        model = tf.keras.model
        hyper_params = dictionary
    """
    final_height, final_width = cfg.IMG_SIZE_HEIGHT, cfg.IMG_SIZE_WIDTH
    img = tf.random.uniform((cfg.BATCH_SIZE, final_height, final_width, 3))
    img_shape = tf.constant([[final_height, final_width]], tf.float32)
    feature_map_shape = cfg.FEATURE_MAP_SHAPE
    total_anchors = feature_map_shape * feature_map_shape * cfg.ANCHOR_COUNT
    gt_boxes = tf.random.uniform((cfg.BATCH_SIZE, 1, 4))
    gt_labels = tf.random.uniform((cfg.BATCH_SIZE, 1), maxval=cfg.NUM_CLASSES, dtype=tf.int32)
    bbox_deltas = tf.random.uniform((cfg.BATCH_SIZE, total_anchors, 4))
    bbox_labels = tf.random.uniform((cfg.BATCH_SIZE, feature_map_shape, feature_map_shape, cfg.ANCHOR_COUNT), maxval=1, dtype=tf.float32)
    if cfg.MASK_REG:
        mask = tf.random.uniform((cfg.BATCH_SIZE, 1, final_height, final_width))
        mask_ids = tf.random.uniform((cfg.BATCH_SIZE, 1), maxval=cfg.NUM_AFFORDANCE_CLASSES, dtype=tf.int32)
        model([img, img_shape, gt_boxes, gt_labels, bbox_deltas, bbox_labels, mask, mask_ids])
    else:
        model([img,img_shape, gt_boxes, gt_labels, bbox_deltas, bbox_labels])


# Training using gradient accumulation
# https://stackoverflow.com/questions/66472201/gradient-accumulation-with-custom-model-fit-in-tf-keras
class CustomModel(Model):

    def __init__(self, loss_names, loss_weights,  **kwargs):
        super().__init__(**kwargs)
        self.n_gradients = tf.constant(2, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)

        for i, layer_name in enumerate(loss_names):
            layer = self.get_layer(layer_name)
            self.add_loss(layer.output * loss_weights[i])
            self.add_metric(layer.output * loss_weights[i], name=layer_name, aggregation="mean")

        # IMPORTANT! this line has to be at the end to avoid listWrapper error
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]

    def train_step(self, data):
        # add 1 to acumulated steps to know when we have to update the gradients
        self.n_acum_step.assign_add(1)

        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Calculate batch gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))


def show_data(feature_extractor, rpn_model, anchors, cfg, mode="training"):
    """Generating rpn model for given backbone base model and hyper params.
    inputs:
        feature_extractor = feature extractor layer from the base model
        rpn_model = tf.keras.model generated rpn model
        anchors = (total_anchors, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        hyper_params = dictionary
        mode = "training" or "inference"

    outputs:
        frcnn_model = tf.keras.model
    """
    input_img = rpn_model.input
    rpn_reg_predictions, rpn_cls_predictions = rpn_model.output
    l2_reg = KR.l2(cfg.WEIGHT_DECAY)
    #
    roi_bboxes = ProposalLayer(anchors, mode, cfg, name="roi_bboxes")([rpn_reg_predictions, rpn_cls_predictions])
    #
    roi_pooled = RoiAlign(cfg, name="roi_pooling")([feature_extractor.output, roi_bboxes])
    #
    output = TimeDistributed(Flatten(), name="frcnn_flatten")(roi_pooled)
    output = TimeDistributed(Dense(4096, activation="relu", kernel_regularizer=l2_reg), name="frcnn_fc1")(output)
    output = TimeDistributed(Dense(4096, activation="relu", kernel_regularizer=l2_reg), name="frcnn_fc2")(output)
    frcnn_cls_predictions = TimeDistributed(Dense(cfg.NUM_CLASSES, activation="softmax", kernel_regularizer=l2_reg),
                                            name="frcnn_cls")(output)
    frcnn_reg_predictions = TimeDistributed(Dense(cfg.NUM_CLASSES * 4, activation="linear", kernel_regularizer=l2_reg),
                                            name="frcnn_reg")(output)

    input_gt_boxes = Input(shape=(None, 4), name="input_gt_boxes", dtype=tf.float32)
    input_gt_labels = Input(shape=(None,), name="input_gt_labels", dtype=tf.int32)
    rpn_cls_actuals = Input(shape=(None, None, cfg.ANCHOR_COUNT), name="input_rpn_cls_actuals", dtype=tf.float32)
    rpn_reg_actuals = Input(shape=(None, 4), name="input_rpn_reg_actuals", dtype=tf.float32)
    input_gt_masks = KL.Input(shape=(None, cfg.IMG_SIZE_HEIGHT, cfg.IMG_SIZE_WIDTH), name="input_gt_masks",
                              dtype=tf.int32)
    input_gt_seg_mask_inds = KL.Input(shape=(None,), name="input_gt_seg_mask_inds", dtype=tf.int32)

    frcnn_reg_actuals, frcnn_cls_actuals, target_maks, rois_pos = ProposalTargetLayer(cfg, name="roi_deltas")(
        [roi_bboxes, input_gt_boxes, input_gt_labels, input_gt_masks, input_gt_seg_mask_inds])

    # only for positive rois
    roi_align_mask = RoiAlign(cfg, name="roi_align_mask")([feature_extractor.output, rois_pos])  # rois_pos])


    pool5_2_conv = KL.TimeDistributed(KL.Conv2D(512, (1, 1), activation="relu", padding="valid",
                                                kernel_initializer=KI.RandomNormal(stddev=0.01),
                                                bias_initializer="zeros",
                                                kernel_regularizer=l2_reg),
                                      name="mask_conv_1")(roi_align_mask)
    pool5_2_conv2 = KL.TimeDistributed(KL.Conv2D(512, (3, 3), activation="relu", padding="same",
                                                 kernel_initializer=KI.RandomNormal(stddev=0.01),
                                                 bias_initializer="zeros",
                                                 kernel_regularizer=l2_reg),
                                       name="mask_conv_2")(pool5_2_conv)

    mask_deconv1 = KL.TimeDistributed(KL.Conv2DTranspose(256, (8, 8), padding="same", strides=(4, 4)),  # , groups=256),
                                      # kernel_initializer=BilinearInitializer(filter_size=8, num_channels_in=512, num_channels_out=256)),
                                      name='mask_deconv_1')(pool5_2_conv2)

    pool5_2_conv3 = KL.TimeDistributed(KL.Conv2D(512, (3, 3), activation="relu", padding="same",
                                                 kernel_initializer=KI.RandomNormal(stddev=0.01),
                                                 bias_initializer="zeros",
                                                 kernel_regularizer=l2_reg),
                                       name="mask_conv_3")(mask_deconv1)

    pool5_2_conv4 = KL.TimeDistributed(KL.Conv2D(512, (3, 3), activation="relu", padding="same",
                                                 kernel_initializer=KI.RandomNormal(stddev=0.01),
                                                 bias_initializer="zeros",
                                                 kernel_regularizer=l2_reg),
                                       name="mask_conv_4")(pool5_2_conv3)

    mask_deconv2 = KL.TimeDistributed(KL.Conv2DTranspose(256, (8, 8), padding="same", strides=(4, 4)),  # , groups=256),
                                      # kernel_initializer=BilinearInitializer(filter_size=8, num_channels_in=512, num_channels_out=256)),
                                      name='mask_deconv_2')(pool5_2_conv4)

    pool5_2_conv5 = KL.TimeDistributed(KL.Conv2D(512, (3, 3), activation="relu", padding="same",
                                                 kernel_initializer=KI.RandomNormal(stddev=0.01),
                                                 bias_initializer="zeros",
                                                 kernel_regularizer=l2_reg),
                                       name="mask_conv_5")(mask_deconv2)

    pool5_2_conv6 = KL.TimeDistributed(KL.Conv2D(512, (3, 3), activation="relu", padding="same",
                                                 kernel_initializer=KI.RandomNormal(stddev=0.01),
                                                 bias_initializer="zeros",
                                                 kernel_regularizer=l2_reg),
                                       name="mask_conv_6")(pool5_2_conv5)

    mask_deconv3 = KL.TimeDistributed(KL.Conv2DTranspose(256, (4, 4), padding="same", strides=(2, 2)),  # , groups=256),
                                      # kernel_initializer=BilinearInitializer(filter_size=4, num_channels_in=512, num_channels_out=256)),
                                      name='mask_deconv_3')(pool5_2_conv6)

    mask_prob_output = KL.TimeDistributed(
        KL.Conv2D(cfg.NUM_AFFORDANCE_CLASSES, (1, 1), padding="valid", activation="softmax",
                  kernel_initializer=KI.RandomNormal(stddev=0.01),
                  bias_initializer="zeros",
                  kernel_regularizer=l2_reg),
        name="mask_score")(mask_deconv3)

    frcnn_model = Model(inputs=[input_img, input_gt_boxes, input_gt_labels, rpn_reg_actuals, rpn_cls_actuals, input_gt_masks, input_gt_seg_mask_inds],
                    outputs=[frcnn_reg_predictions, frcnn_cls_predictions, frcnn_reg_actuals, frcnn_cls_actuals, roi_bboxes, target_maks, rois_pos])

    return frcnn_model

# WITH RESIZING
# class ProposalLayer(Layer):
#     """Generating bounding boxes from rpn predictions.
#     First calculating the boxes from predicted deltas and label probs.
#     Then applied non max suppression and selecting "train or test nms_topn" boxes.
#     inputs:
#         rpn_bbox_deltas = (batch_size, img_output_height, img_output_width, anchor_count * [delta_y, delta_x, delta_h, delta_w])
#             img_output_height and img_output_width are calculated to the base model feature map
#         rpn_labels = (batch_size, img_output_height, img_output_width, anchor_count)
#
#     outputs:
#         roi_bboxes = (batch_size, train/test_nms_topn, [y1, x1, y2, x2])
#     """
#
#     def __init__(self, anchors, mode, cfg, **kwargs):
#         super(ProposalLayer, self).__init__(**kwargs)
#         self.cfg = cfg
#         self.mode = mode
#         self.anchors = tf.constant(anchors, dtype=tf.float32)
#
#     def get_config(self):
#         config = super(ProposalLayer, self).get_config()
#         config.update({"cfg": self.cfg, "anchors": self.anchors, "mode": self.mode})
#         return config
#
#     def call(self, inputs):
#         rpn_bbox_deltas = inputs[0]
#         rpn_labels = inputs[1]
#         anchors = self.anchors
#
#         pre_nms_topn = self.cfg.PRE_NMS_TOPN
#         post_nms_topn = self.cfg.TRAIN_NMS_TOPN if self.mode == "training" else self.cfg.TEST_NMS_TOPN
#         nms_iou_threshold = self.cfg.NMS_IOU_THRESHOLD
#         variances = self.cfg.VARIANCES
#         total_anchors = anchors.shape[0]
#         batch_size = tf.shape(rpn_bbox_deltas)[0]
#         rpn_bbox_deltas = tf.reshape(rpn_bbox_deltas, (batch_size, total_anchors, 4))
#         rpn_labels = tf.reshape(rpn_labels, (batch_size, total_anchors))
#
#         rpn_bbox_deltas *= variances
#         rpn_bboxes = bbox_utils.get_bboxes_from_deltas(anchors, rpn_bbox_deltas)
#
#         _, pre_indices = tf.nn.top_k(rpn_labels, pre_nms_topn)
#
#         pre_roi_bboxes = tf.gather(rpn_bboxes, pre_indices, batch_dims=1)
#         pre_roi_labels = tf.gather(rpn_labels, pre_indices, batch_dims=1)
#
#         pre_roi_bboxes = tf.reshape(pre_roi_bboxes, (batch_size, pre_nms_topn, 1, 4))
#         pre_roi_labels = tf.reshape(pre_roi_labels, (batch_size, pre_nms_topn, 1))
#
#         roi_bboxes, _, _, _ = bbox_utils.non_max_suppression(pre_roi_bboxes, pre_roi_labels,
#                                                           max_output_size_per_class=post_nms_topn,
#                                                           max_total_size=post_nms_topn,
#                                                           iou_threshold=nms_iou_threshold)
#         return tf.stop_gradient(roi_bboxes)

# WITH RESIZING
# class ProposalTargetLayer(Layer):
#     """Calculating faster rcnn actual bounding box deltas and labels.
#     This layer only running on the training phase.
#     inputs:
#         roi_bboxes = (batch_size, nms_topn, [y1, x1, y2, x2])
#         gt_boxes = (batch_size, padded_gt_boxes_size, [y1, x1, y2, x2])
#         gt_labels = (batch_size, padded_gt_boxes_size)
#         gt_masks = (batch_size, num_masks, img_height, img_width)
#
#     outputs:
#         roi_bbox_deltas = (batch_size, train_nms_topn * total_labels, [delta_y, delta_x, delta_h, delta_w])
#         roi_bbox_labels = (batch_size, train_nms_topn, total_labels)
#     """
#
#     def __init__(self, cfg, **kwargs):
#         super(ProposalTargetLayer, self).__init__(**kwargs)
#         self.cfg = cfg
#         # self.img_height = tf.cast(img_height, tf.float32)
#         # self.img_width = tf.cast(img_width, tf.float32)
#
#     def get_config(self):
#         config = super(ProposalTargetLayer, self).get_config()
#         config.update({"cfg": self.cfg, "img_height": self.img_height, "img_width": self.img_width})
#         return config
#
#     def call(self, inputs):
#         roi_bboxes = inputs[0]
#         gt_boxes = inputs[1]
#         gt_labels = inputs[2]
#         if self.cfg.MASK_REG:
#             gt_masks = inputs[3]
#             gt_seg_mask_inds = inputs[4]
#
#         total_labels = self.cfg.NUM_CLASSES
#         # total_pos_bboxes = int(self.cfg.RPN_BATCHSIZE * self.cfg.RPN_FG_FRACTION)
#         # total_neg_bboxes = self.cfg.RPN_BATCHSIZE - total_pos_bboxes
#         # TODO: try to increment number of positive rois
#         # Negative ROIs. Add enough to maintain positive:negative ratio.
#         r = 1.0 / self.cfg.ROI_POSITIVE_RATIO
#         total_pos_bboxes = int(self.cfg.TRAIN_ROIS_PER_IMAGE * self.cfg.ROI_POSITIVE_RATIO)
#         total_neg_bboxes = int(r * float(total_pos_bboxes)) - total_pos_bboxes
#
#         variances = self.cfg.VARIANCES
#         # batch_size, total_bboxes = tf.shape(roi_bboxes)[0], tf.shape(roi_bboxes)[1]
#
#         # Calculate iou values between each bboxes and ground truth boxes
#         iou_map, _ = bbox_utils.generate_iou_map(roi_bboxes, gt_boxes)
#         # Get max index value for each row
#         max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
#         # IoU map has iou values for every gt boxes and we merge these values column wise
#         merged_iou_map = tf.reduce_max(iou_map, axis=2)
#         #
#         pos_mask = tf.greater(merged_iou_map, self.cfg.TRAIN_FG_THRES)
#         pos_mask = train_utils.randomly_select_xyz_mask(pos_mask, tf.constant([total_pos_bboxes], dtype=tf.int32))
#
#         neg_mask = tf.logical_and(tf.less(merged_iou_map, self.cfg.TRAIN_BG_THRESH_HI), tf.greater(merged_iou_map, self.cfg.TRAIN_BG_THRESH_LO))
#         neg_mask = train_utils.randomly_select_xyz_mask(neg_mask, tf.constant([total_neg_bboxes], dtype=tf.int32))
#
#         # take corresponding gt boxes and gt labels to rois
#         gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
#         expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, axis=-1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
#
#         gt_labels_map = tf.gather(gt_labels, max_indices_each_gt_box, batch_dims=1)
#         pos_gt_labels = tf.where(pos_mask, gt_labels_map, tf.constant(-1, dtype=tf.int32))
#         neg_gt_labels = tf.cast(neg_mask, dtype=tf.int32)
#         expanded_gt_labels = tf.cast(pos_gt_labels + neg_gt_labels, dtype=tf.int32)
#         # (batch_size, num_rois, 4)
#         roi_bbox_deltas = bbox_utils.get_deltas_from_bboxes(roi_bboxes, expanded_gt_boxes) / variances
#
#         # roi_bbox_labels = expanded_gt_labels
#
#         # Transform to one hot representation (batch_size, num_rois, num_classes)
#         roi_bbox_labels = tf.one_hot(expanded_gt_labels, total_labels)
#         # scatter_indices = tf.tile(tf.expand_dims(roi_bbox_labels, -1), (1, 1, 1, 4))
#         # roi_bbox_deltas = scatter_indices * tf.expand_dims(roi_bbox_deltas, -2)
#         # roi_bbox_deltas = tf.reshape(roi_bbox_deltas, (batch_size, total_bboxes * total_labels, 4))
#
#         if self.cfg.MASK_REG:
#             # Take only positive rois for mask training and corresponding roi_gt_boxes
#             pos_indices = tf.where(pos_mask)
#             positive_count = tf.shape(pos_indices)[0]
#             positive_rois = tf.gather_nd(roi_bboxes, pos_indices)
#             roi_gt_boxes = tf.gather_nd(gt_boxes_map, pos_indices)
#
#             y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
#             y1t, x1t, y2t, x2t = tf.split(roi_gt_boxes, 4, axis=1)
#
#             img_w = float(self.cfg.IMG_SIZE_WIDTH)
#             img_h = float(self.cfg.IMG_SIZE_HEIGHT)
#             # sanity check
#             x1 = tf.minimum(img_w - 1, tf.maximum(0.0, x1))
#             y1 = tf.minimum(img_h - 1, tf.maximum(0.0, y1))
#             x2 = tf.minimum(img_w - 1, tf.maximum(0.0, x2))
#             y2 = tf.minimum(img_h - 1, tf.maximum(0.0, y2))
#             x1t = tf.minimum(img_w - 1, tf.maximum(0.0, x1t))
#             y1t = tf.minimum(img_h - 1, tf.maximum(0.0, y1t))
#             x2t = tf.minimum(img_w - 1, tf.maximum(0.0, x2t))
#             y2t = tf.minimum(img_h - 1, tf.maximum(0.0, y2t))
#
#             w = (x2 - x1) + 1
#             h = (y2 - y1) + 1
#
#             # compute overlap between roi coordinate and gt_roi coordinate TODO: use overlap function?
#             x1o = tf.maximum(x1, x1t)
#             y1o = tf.maximum(y1, y1t)
#             x2o = tf.minimum(x2, x2t)
#             y2o = tf.minimum(y2, y2t)
#
#             if positive_count != 0:
#                 # Calculate labels in original mask -> gt_masks=(batch_size, num_masks, img_height, img_width)
#                 original_affordance_labels = tf.unique(tf.reshape(gt_masks, [-1]))
#                 original_affordance_labels = tf.sort(original_affordance_labels.y)
#
#                 # filter indices of gt boxes
#                 indices_pos_gt_boxes = tf.boolean_mask(max_indices_each_gt_box, pos_mask)
#
#                 # mask associated wrt to true bbox (batch_size, positive_rois, mask_size, mask_size)
#                 gt_mask = tf.gather(gt_masks, indices_pos_gt_boxes, axis=1)
#
#                 gt_mask = tf.cast(tf.expand_dims(gt_mask, axis=4), tf.float32)
#                 y1o = tf.squeeze(y1o, axis=1)
#                 x1o = tf.squeeze(x1o, axis=1)
#                 y2o = tf.squeeze(y2o, axis=1)
#                 x2o = tf.squeeze(x2o, axis=1)
#
#                 # create boxes to crop and indexes where each mask has its own box
#                 boxes = tf.cast(tf.stack([y1o, x1o, y2o, x2o], axis=1), tf.float32)
#                 box_index = tf.range(positive_count)
#
#                 # remove batch dim -> needed for crop and resize op
#                 gt_mask = tf.squeeze(gt_mask, axis=0)
#
#                 # crop and resize the masks individually
#                 positive_masks = self._crop_and_resize_masks(gt_mask, boxes, positive_rois, positive_count, original_affordance_labels)
#
#                 # Add batch dim
#                 positive_masks = tf.expand_dims(positive_masks, axis=0)
#                 positive_rois = tf.expand_dims(positive_rois, axis=0)
#                 masks = positive_masks  # tf.concat([positive_masks, negative_masks], axis=0)
#             else:
#                 positive_rois = tf.expand_dims(positive_rois, axis=0)
#                 masks = tf.constant(0, dtype=tf.int32, shape=[1, 0, self.cfg.TRAIN_MASK_SIZE, self.cfg.TRAIN_MASK_SIZE])
#
#             return tf.stop_gradient(roi_bbox_deltas), tf.stop_gradient(roi_bbox_labels), tf.stop_gradient(masks), \
#                 tf.stop_gradient(positive_rois)
#
#         return tf.stop_gradient(roi_bbox_deltas), tf.stop_gradient(roi_bbox_labels)
#
#     def _crop_and_resize_masks(self, masks, overlapping_boxes, rois, positive_count, original_aff_labels):
#         # overlapping_boxes = tf.cast(bbox_utils.denormalize_bboxes(overlapping_boxes, self.img_height, self.img_width), tf.int32)
#         # rois = tf.cast(bbox_utils.denormalize_bboxes(rois, self.img_height, self.img_width), tf.int32)
#         overlapping_boxes = tf.cast(bbox_utils.denormalize_bboxes(overlapping_boxes, self.cfg.IMG_SIZE_HEIGHT, self.cfg.IMG_SIZE_WIDTH), tf.int32)
#         rois = tf.cast(bbox_utils.denormalize_bboxes(rois, self.cfg.IMG_SIZE_HEIGHT, self.cfg.IMG_SIZE_WIDTH), tf.int32)
#
#         # overlapping_boxes = tf.cast(overlapping_boxes, tf.int32)
#         # rois = tf.cast(rois, tf.int32)
#
#         num_masks = tf.shape(masks)[0]
#         final_masks = tf.zeros((num_masks, self.cfg.TRAIN_MASK_SIZE, self.cfg.TRAIN_MASK_SIZE))
#         for i in range(num_masks):
#             mask = masks[i]
#
#             # get roi and overlap area coordinates
#             y1, x1, y2, x2 = tf.split(rois[i], 4, axis=0)
#             y1, x1, y2, x2 = tf.squeeze(y1), tf.squeeze(x1), tf.squeeze(y2), tf.squeeze(x2)
#
#             y1o, x1o, y2o, x2o = tf.split(overlapping_boxes[i], 4, axis=0)
#             y1o, x1o, y2o, x2o = tf.squeeze(y1o), tf.squeeze(x1o), tf.squeeze(y2o), tf.squeeze(x2o)
#
#             # take overlap area between gt_bbox and roi
#             overlapping_mask_area = mask[y1o:y2o, x1o:x2o]
#
#             # calculate offsets with 0 above and in the left of the overlapping area
#             offset_height = y1o - y1
#             offset_width = x1o - x1
#
#             # calculate roi height and width
#             target_height = y2 - y1 + 1
#             target_width = x2 - x1 + 1
#
#             roi_mask = tf.image.pad_to_bounding_box(overlapping_mask_area, offset_height, offset_width, target_height, target_width)
#
#             # # add overlapping area inside the roi and resize to mask size
#             # roi_mask[(y1o - y1):(y2o - y1), (x1o - x1):(x2o - x1)] = overlapping_mask_area
#             roi_mask = tf.image.resize(roi_mask, [self.cfg.TRAIN_MASK_SIZE, self.cfg.TRAIN_MASK_SIZE], method='bilinear')
#
#             # Create a structure with 0 for all masks except for the current mask and add that to the final mask structure
#             temp_masks = tf.scatter_nd([[i]], tf.expand_dims(tf.squeeze(roi_mask, axis=2), axis=0),
#                                        [num_masks, self.cfg.TRAIN_MASK_SIZE, self.cfg.TRAIN_MASK_SIZE])
#             final_masks += temp_masks
#
#         final_masks = _convert_mask_to_original_ids_manual(positive_count, final_masks, original_aff_labels, self.cfg.TRAIN_MASK_SIZE)
#         return final_masks


# class BilinearInitializer(KI.Initializer):
#     """
#     Create weights matrix for transposed convolution with bilinear filter initialization (as in Caffe)
#     https://github.com/warmspringwinds/tensorflow_notes/blob/master/upsampling_segmentation.ipynb
#     https://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/initializers.html#deconv2d_bilinear_upsampling_initializer
#     """
#     def __init__(self, filter_size, num_channels_in, num_channels_out):
#       self.filter_size = filter_size
#       self.num_channels_in = num_channels_in
#       self.num_channels_out = num_channels_out
#
#     def get_config(self):  # To support serialization
#         config = super(BilinearInitializer, self).get_config()
#         config.update({"filter_size": self.filter_size,
#                        "num_channels_in": self.num_channels_in,
#                        "num_channels_out": self.num_channels_out})
#         return config
#
#     def __call__(self, shape, dtype=None):
#         upsample_kernel = self.upsample_filt()
#
#         total_filter_size = self.filter_size * self.filter_size     # if we take into account filters in rows instead of matrices
#         upsample_kernel = tf.reshape(upsample_kernel, [total_filter_size])
#         index_kernel = tf.range(total_filter_size)      # indices for each position in the filter (in 1 row)
#         indices_diagonals = tf.repeat(index_kernel, tf.fill([total_filter_size], self.num_channels_out))     # indices for each value in diagonal
#         indices_diagonals = tf.reshape(indices_diagonals, [total_filter_size, self.num_channels_out])
#         # substitute indices in diagonal for each corresponding value and reshape to (filter_size, filter_size, num_classes, num_classes)
#         diagonal_values = tf.gather(upsample_kernel, indices_diagonals)
#         diagonal_matrices = tf.linalg.diag(diagonal_values)
#         complement_to_diag = tf.zeros(diagonal_matrices.shape, dtype=tf.float32)
#         weights = tf.concat([diagonal_matrices, complement_to_diag], axis=2)
#         weights = tf.reshape(weights, [self.filter_size, self.filter_size, self.num_channels_out, self.num_channels_in])
#         return weights
#
#     def upsample_filt(self):
#         """
#         Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
#         """
#         factor = (self.filter_size + 1) // 2
#         if self.filter_size % 2 == 1:
#             center = factor - 1
#         else:
#             center = factor - 0.5
#
#         g = tf.meshgrid(tf.range(self.filter_size), tf.range(self.filter_size))
#         a = tf.cast(tf.reshape(g[0][0], [self.filter_size, 1]), tf.float32)
#         b = tf.cast(tf.reshape(g[1][:,0], [1, self.filter_size]), tf.float32)
#         return (1 - tf.abs(a - center)) / factor * (1 - tf.abs(b - center)) / factor

# add group, reg and weights --> no resize
# def get_model_only_with_masks(feature_extractor, rpn_model, anchors, cfg, mode="training"):
#     """Generating rpn model for given backbone base model and hyper params.
#     inputs:
#         feature_extractor = feature extractor layer from the ase model
#         rpn_model = tf.keras.model generated rpn model
#         anchors = (total_anchors, [y1, x1, y2, x2])
#             these values in normalized format between [0, 1]
#         hyper_params = dictionary
#         mode = "training" or "inference"
#
#     outputs:
#         frcnn_model = tf.keras.model
#     """
#     input_img = rpn_model.input
#     rpn_reg_predictions, rpn_cls_predictions = rpn_model.output
#     l2_reg = KR.l2(cfg.WEIGHT_DECAY)
#     #
#     roi_bboxes = ProposalLayer(anchors, mode, cfg, name="roi_bboxes")([rpn_reg_predictions, rpn_cls_predictions])
#     #
#     roi_pooled = RoiAlign(cfg, name="roi_pooling")([feature_extractor.output, roi_bboxes])
#     #
#     output = TimeDistributed(Flatten(), name="frcnn_flatten")(roi_pooled)
#     output = TimeDistributed(Dense(4096, activation="relu", kernel_regularizer=l2_reg), name="frcnn_fc1")(output)
#     output = TimeDistributed(Dense(4096, activation="relu", kernel_regularizer=l2_reg), name="frcnn_fc2")(output)
#     frcnn_cls_predictions = TimeDistributed(Dense(cfg.NUM_CLASSES, activation="softmax", kernel_regularizer=l2_reg), name="frcnn_cls")(output)
#     frcnn_reg_predictions = TimeDistributed(Dense(cfg.NUM_CLASSES * 4, activation="linear", kernel_regularizer=l2_reg), name="frcnn_reg")(output)
#
#     if mode == "training":
#         input_gt_boxes = Input(shape=(None, 4), name="input_gt_boxes", dtype=tf.float32)
#         input_gt_labels = Input(shape=(None, ), name="input_gt_labels", dtype=tf.int32)
#         rpn_cls_actuals = Input(shape=(None, None, cfg.ANCHOR_COUNT), name="input_rpn_cls_actuals", dtype=tf.float32)
#         rpn_reg_actuals = Input(shape=(None, 4), name="input_rpn_reg_actuals", dtype=tf.float32)
#         input_gt_masks = KL.Input(shape=(None, cfg.IMG_SIZE_HEIGHT, cfg.IMG_SIZE_WIDTH), name="input_gt_masks", dtype=tf.int32)
#         input_gt_seg_mask_inds = KL.Input(shape=(None,), name="input_gt_seg_mask_inds", dtype=tf.int32)
#
#         frcnn_reg_actuals, frcnn_cls_actuals, target_maks, rois_pos = ProposalTargetLayer(cfg, name="roi_deltas")(
#             [roi_bboxes, input_gt_boxes, input_gt_labels, input_gt_masks, input_gt_seg_mask_inds])
#
#         # only for positive rois
#         roi_align_mask = RoiAlign(cfg, name="roi_align_mask")([feature_extractor.output, rois_pos]) #rois_pos])
#
#     else:
#         bboxes, labels, scores = Decoder(cfg.VARIANCES, cfg.NUM_CLASSES, nms_threshold=cfg.TEST_NMS_THRESHOLD,
#                                          max_total_size=cfg.MAX_PER_IMAGE, score_threshold=cfg.SCORE_THRESHOLD,
#                                          name="faster_rcnn_decoder")(
#             [roi_bboxes, frcnn_reg_predictions, frcnn_cls_predictions])
#         # TODO: check if we have to use pred_bboxes or we have to calculate different rois for this
#         roi_align_mask = RoiAlign(cfg, name="roi_align_mask")([feature_extractor.output, bboxes])
#
#     pool5_2_conv = KL.TimeDistributed(KL.Conv2D(512, (1, 1), activation="relu", padding="valid",
#                                                 kernel_initializer=KI.RandomNormal(stddev=0.01),
#                                                 bias_initializer="zeros",
#                                                 kernel_regularizer=l2_reg),
#                                       name="mask_conv_1")(roi_align_mask)
#     pool5_2_conv2 = KL.TimeDistributed(KL.Conv2D(512, (3, 3), activation="relu", padding="same",
#                                                  kernel_initializer=KI.RandomNormal(stddev=0.01),
#                                                  bias_initializer="zeros",
#                                                  kernel_regularizer=l2_reg),
#                                        name="mask_conv_2")(pool5_2_conv)
#
#     mask_deconv1 = KL.TimeDistributed(KL.Conv2DTranspose(256, (8, 8), padding="same", strides=(4, 4), groups=256, kernel_regularizer=l2_reg),
#                                                          # kernel_initializer=BilinearInitializer(filter_size=8, num_channels_in=512, num_channels_out=256)),
#                                       name='mask_deconv_1')(pool5_2_conv2)
#
#     pool5_2_conv3 = KL.TimeDistributed(KL.Conv2D(512, (3, 3), activation="relu", padding="same",
#                                                  kernel_initializer=KI.RandomNormal(stddev=0.01),
#                                                  bias_initializer="zeros",
#                                                  kernel_regularizer=l2_reg),
#                                        name="mask_conv_3")(mask_deconv1)
#
#     pool5_2_conv4 = KL.TimeDistributed(KL.Conv2D(512, (3, 3), activation="relu", padding="same",
#                                                  kernel_initializer=KI.RandomNormal(stddev=0.01),
#                                                  bias_initializer="zeros",
#                                                  kernel_regularizer=l2_reg),
#                                        name="mask_conv_4")(pool5_2_conv3)
#
#     mask_deconv2 = KL.TimeDistributed(KL.Conv2DTranspose(256, (8, 8), padding="same", strides=(4, 4), groups=256, kernel_regularizer=l2_reg),
#                                                          # kernel_initializer=BilinearInitializer(filter_size=8, num_channels_in=512, num_channels_out=256)),
#                                       name='mask_deconv_2')(pool5_2_conv4)
#
#     pool5_2_conv5 = KL.TimeDistributed(KL.Conv2D(512, (3, 3), activation="relu", padding="same",
#                                                  kernel_initializer=KI.RandomNormal(stddev=0.01),
#                                                  bias_initializer="zeros",
#                                                  kernel_regularizer=l2_reg),
#                                        name="mask_conv_5")(mask_deconv2)
#
#     pool5_2_conv6 = KL.TimeDistributed(KL.Conv2D(512, (3, 3), activation="relu", padding="same",
#                                                  kernel_initializer=KI.RandomNormal(stddev=0.01),
#                                                  bias_initializer="zeros",
#                                                  kernel_regularizer=l2_reg),
#                                        name="mask_conv_6")(pool5_2_conv5)
#
#     mask_deconv3 = KL.TimeDistributed(KL.Conv2DTranspose(256, (4, 4), padding="same", strides=(2, 2), groups=256, kernel_regularizer=l2_reg),
#                                                          # kernel_initializer=BilinearInitializer(filter_size=4, num_channels_in=512, num_channels_out=256)),
#                                       name='mask_deconv_3')(pool5_2_conv6)
#
#     mask_prob_output = KL.TimeDistributed(KL.Conv2D(cfg.NUM_AFFORDANCE_CLASSES, (1, 1), padding="valid", activation="softmax",
#                                                     kernel_initializer=KI.RandomNormal(stddev=0.01),
#                                                     bias_initializer="zeros",
#                                                     kernel_regularizer=l2_reg),
#                                           name="mask_score")(mask_deconv3)
#
#     if mode == "training":
#         rpn_cls_loss_layer = Lambda(train_utils.rpn_cls_loss, name=cfg.LOSS_NAMES[1])([rpn_cls_actuals, rpn_cls_predictions])
#         rpn_reg_loss_layer = Lambda(train_utils.rpn_reg_loss, name=cfg.LOSS_NAMES[0])([rpn_reg_actuals, rpn_reg_predictions, rpn_cls_actuals])
#         reg_loss_layer = Lambda(train_utils.reg_loss, name=cfg.LOSS_NAMES[2])([frcnn_reg_actuals, frcnn_reg_predictions, frcnn_cls_actuals])
#         cls_loss_layer = Lambda(train_utils.cls_loss, name=cfg.LOSS_NAMES[3])([frcnn_cls_actuals, frcnn_cls_predictions])
#         mask_loss_layer = KL.Lambda(train_utils.affordance_mask_loss, name=cfg.LOSS_NAMES[4])([target_maks, mask_prob_output])
#
#         frcnn_model = Model(inputs=[input_img, input_gt_boxes, input_gt_labels, rpn_reg_actuals, rpn_cls_actuals, input_gt_masks, input_gt_seg_mask_inds],
#                         outputs=[rpn_reg_loss_layer, rpn_cls_loss_layer, reg_loss_layer, cls_loss_layer, mask_loss_layer])
#
#
#         for i, layer_name in enumerate(cfg.LOSS_NAMES):
#             layer = frcnn_model.get_layer(layer_name)
#             frcnn_model.add_loss(layer.output * cfg.LOSS_WEIGHTS[i])
#             frcnn_model.add_metric(layer.output * cfg.LOSS_WEIGHTS[i], name=layer_name, aggregation="mean")
#     else:
#         frcnn_model = Model(inputs=[input_img], outputs=[bboxes, labels, scores, mask_prob_output])
#     return frcnn_model
