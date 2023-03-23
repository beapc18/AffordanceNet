import tensorflow as tf
import numpy as np
from utils import bbox_utils, drawing_utils
from scipy.ndimage import distance_transform_edt, gaussian_filter
import cv2


def init_stats(labels):
    """
        Initializes the statistics.
    :param labels: array with all possible labels
    :returns: json with empty statistics for all labels
    """
    stats = {}
    for i, label in enumerate(labels):
        if i == 0:
            continue
        stats[i] = {
            "label": label,
            "total": 0,
            "tp": [],
            "fp": [],
            "scores": [],
            "q": []
        }
    return stats


def update_stats(obj_stats, gt_bboxes, gt_labels, pred_bboxes, pred_labels, pred_scores, mask_eval=False,
                 affordance_stats=None, gt_masks=None, pred_masks=None, img_height=None, img_width=None, iou_thres=0.3):
    """
        Updates statistics for object classification and affordance detection.
    :param obj_stats: accumulated statistics for object classification
    :param gt_bboxes: ground truth normalized bounding boxes    (batch_size, num_gt_bboxes, 4)
    :param gt_labels: ground truth labels for gt_boxes          (batch_size, num_gt_bboxes)
    :param pred_bboxes: predicted normalized bounding boxes     (batch_size, num_pred_bboxes, 4)
    :param pred_labels: predicted labels for pred_bboxes        (batch_size, num_pred_bboxes)
    :param pred_scores: predicted scores for pred_bboxes        (batch_size, num_pred_bboxes)
    :param mask_eval: True if there are predicted masks, False otherwise
    :param affordance_stats: accumulated statistics for affordance evaluation
    :param gt_masks: ground truth masks                                             (batch_size, num_gt_bboxes, orig_mask_height, orig_mask_width)
    :param pred_masks: predicted masks with prob for each pixel for each class      (batch_size, num_pred_bboxes, train_mask_size, train_mask_size, num_affordance_classes)
    :param img_height: image height
    :param img_width: image width
    :returns: jsons with updated statistics for object classification and affordance detection
    """
    # create empty mask to accumulate masks for all bboxes in one single mask
    final_gt_mask = np.zeros((img_height, img_width))
    final_pred_mask = np.zeros((img_height, img_width))

    # iou for each pred_bbox wrt each gt_box
    iou_map, zero_iou = bbox_utils.generate_iou_map(pred_bboxes, gt_bboxes)

    # update stats only if there are some iou that are not 0
    if not zero_iou:
        # take max iou for each pred_bbox and its corresponding gt_box indices
        merged_iou_map = tf.reduce_max(iou_map, axis=-1)
        max_indices_each_gt = tf.argmax(iou_map, axis=-1, output_type=tf.int32)
        sorted_ids = tf.argsort(merged_iou_map, direction="DESCENDING")

        # Add total of true labels for each class to stats
        count_holder = tf.unique_with_counts(tf.reshape(gt_labels, (-1,)))
        for i, gt_label in enumerate(count_holder[0]):
            if gt_label == -1:
                continue
            gt_label = int(gt_label)
            obj_stats[gt_label]["total"] += int(count_holder[2][i])

        for batch_id, m in enumerate(merged_iou_map):
            true_labels = []
            for i, sorted_id in enumerate(sorted_ids[batch_id]):
                pred_label = pred_labels[batch_id, sorted_id]
                if pred_label == 0:
                    continue
                iou = merged_iou_map[batch_id, sorted_id]
                gt_id = max_indices_each_gt[batch_id, sorted_id]
                gt_label = int(gt_labels[batch_id, gt_id])
                pred_label = int(pred_label)
                score = pred_scores[batch_id, sorted_id]
                obj_stats[pred_label]["scores"].append(score)
                obj_stats[pred_label]["tp"].append(0)
                obj_stats[pred_label]["fp"].append(0)

                # correct detection
                if iou >= iou_thres and pred_label == gt_label and gt_id not in true_labels:
                    obj_stats[pred_label]["tp"][-1] = 1
                    true_labels.append(gt_id)
                    if mask_eval:
                        final_gt_mask, final_pred_mask = update_final_masks(final_gt_mask, final_pred_mask, gt_bboxes[batch_id, gt_id],
                                                                    gt_masks[batch_id, gt_id].numpy(), pred_masks[batch_id, sorted_id],
                                                                    img_height, img_width)
                else:
                    obj_stats[pred_label]["fp"][-1] = 1
        if mask_eval:
            affordance_stats = update_stats_affordances(affordance_stats, final_gt_mask, final_pred_mask)
    return obj_stats, affordance_stats


def update_stats_objecness(obj_stats, gt_bboxes, gt_labels, pred_bboxes, pred_labels, pred_scores, mask_eval=False,
                 affordance_stats=None, gt_masks=None, pred_masks=None, img_height=None, img_width=None, iou_thres=0.3):
    """
        Updates statistics for object classification and affordance detection.
    :param obj_stats: accumulated statistics for object classification
    :param gt_bboxes: ground truth normalized bounding boxes    (batch_size, num_gt_bboxes, 4)
    :param gt_labels: ground truth labels for gt_boxes          (batch_size, num_gt_bboxes)
    :param pred_bboxes: predicted normalized bounding boxes     (batch_size, num_pred_bboxes, 4)
    :param pred_labels: predicted labels for pred_bboxes        (batch_size, num_pred_bboxes)
    :param pred_scores: predicted scores for pred_bboxes        (batch_size, num_pred_bboxes)
    :param mask_eval: True if there are predicted masks, False otherwise
    :param affordance_stats: accumulated statistics for affordance evaluation
    :param gt_masks: ground truth masks                                             (batch_size, num_gt_bboxes, orig_mask_height, orig_mask_width)
    :param pred_masks: predicted masks with prob for each pixel for each class      (batch_size, num_pred_bboxes, train_mask_size, train_mask_size, num_affordance_classes)
    :param img_height: image height
    :param img_width: image width
    :returns: jsons with updated statistics for object classification and affordance detection
    """
    # create empty mask to accumulate masks for all bboxes in one single mask
    final_gt_mask = np.zeros((img_height, img_width))
    final_pred_mask = np.zeros((img_height, img_width))

    # iou for each pred_bbox wrt each gt_box
    iou_map, zero_iou = bbox_utils.generate_iou_map(pred_bboxes, gt_bboxes)

    # update stats only if there are some iou that are not 0
    if not zero_iou:
        # take max iou for each pred_bbox and its corresponding gt_box indices
        merged_iou_map = tf.reduce_max(iou_map, axis=-1)
        max_indices_each_gt = tf.argmax(iou_map, axis=-1, output_type=tf.int32)
        sorted_ids = tf.argsort(merged_iou_map, direction="DESCENDING")

        # Add total of true labels for each class to stats
        count_holder = tf.unique_with_counts(tf.reshape(gt_labels, (-1,)))
        for i, gt_label in enumerate(count_holder[0]):
            if gt_label == -1:
                continue
            # gt_label = int(gt_label)
            if int(gt_label) > 0:
                gt_label = 1
            obj_stats[gt_label]["total"] += int(count_holder[2][i])

        for batch_id, m in enumerate(merged_iou_map):
            true_labels = []
            for i, sorted_id in enumerate(sorted_ids[batch_id]):
                pred_label = pred_labels[batch_id, sorted_id]
                if pred_label == 0:
                    continue
                iou = merged_iou_map[batch_id, sorted_id]
                gt_id = max_indices_each_gt[batch_id, sorted_id]
                gt_label = int(gt_labels[batch_id, gt_id])
                pred_label = int(pred_label)
                score = pred_scores[batch_id, sorted_id]
                obj_stats[pred_label]["scores"].append(score)
                obj_stats[pred_label]["tp"].append(0)
                obj_stats[pred_label]["fp"].append(0)

                if int(gt_label) > 0:
                    gt_label = 1

                # correct detection
                if iou >= iou_thres and pred_label == gt_label and gt_id not in true_labels:
                    obj_stats[pred_label]["tp"][-1] = 1
                    true_labels.append(gt_id)
                    if mask_eval:
                        final_gt_mask, final_pred_mask = update_final_masks(final_gt_mask, final_pred_mask, gt_bboxes[batch_id, gt_id],
                                                                    gt_masks[batch_id, gt_id].numpy(), pred_masks[batch_id, sorted_id],
                                                                    img_height, img_width)
                else:
                    obj_stats[pred_label]["fp"][-1] = 1
        if mask_eval:
            affordance_stats = update_stats_affordances(affordance_stats, final_gt_mask, final_pred_mask)
    return obj_stats, affordance_stats

def update_final_masks(final_gt_mask, final_pred_mask, gt_bbox, gt_mask, pred_mask, img_height, img_width):
    """
        Updates final masks, ground truth and predicted, to add each object mask to one final mask for the whole image.
    :param final_gt_mask: mask with all gt masks in the image
    :param final_pred_mask: mask with all predicted masks in the image
    :param gt_bbox: ground truth normalized bounding box
    :param gt_mask: ground truth mask
    :param pred_mask: predicted mask contains for each pixel the probability of each class
    :param img_height: image height
    :param img_width: image width
    :returns: final masks for gt and predicted masks including masks for new objects
    """
    denormalized_bboxes = bbox_utils.denormalize_bboxes(gt_bbox, img_height, img_width)

    # crop only bbox area and add it to final gt_mask
    y1, x1, y2, x2 = tf.split(denormalized_bboxes, 4)
    y1, x1, y2, x2 = y1.numpy().astype(int)[0], x1.numpy().astype(int)[0], y2.numpy().astype(int)[0], x2.numpy().astype(int)[0]
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    gt_mask = gt_mask[y1:y2, x1:x2]
    final_gt_mask[y1:y2, x1:x2] = gt_mask

    # obtain correct affordance label for each position in the mask (max index for each position in the mask)
    pred_mask = np.argmax(pred_mask, axis=2)

    # resize predicted mask to gt_mask size and convert to original affordance label ids
    original_affordance_labels = np.unique(pred_mask)
    pred_mask = drawing_utils.reset_mask_ids(pred_mask, original_affordance_labels)
    pred_mask = cv2.resize(pred_mask.astype('float'), (bbox_width, bbox_height), interpolation=cv2.INTER_LINEAR)
    pred_mask = drawing_utils.convert_mask_to_original_ids_manual(pred_mask, original_affordance_labels)
    # add mask values to current mask but preserving the maximum index -> 0 less importance, 10 max importance
    provisional_mask = final_pred_mask[y1:y2, x1:x2]
    final_pred_mask[y1:y2, x1:x2] = np.maximum(pred_mask, provisional_mask)
    return final_gt_mask, final_pred_mask


def update_stats_affordances(stats, final_gt_mask, final_pred_mask):
    """
        Updates stats for affordance detection.
        Calculates FwB measurement for the whole image that contains all the masks for all the objects.
        https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/FGEval/resources/WFb.m
    :param stats: accumulated statistics for affordance evaluation
    :param final_gt_mask: mask with all gt masks in the image
    :param final_pred_mask: mask with all predicted masks in the image
    :returns: json with updated stats for affordance detection
    """
    ids = np.unique(final_gt_mask)
    ids = np.delete(ids, np.where(ids == 0)) # remove id 0 if it exists -> ignore BG
    eps = np.finfo(float).eps

    for id in ids:
        # separate BG and FG in gt mask and predicted mask (0 for bg, 1 for class)
        G = final_gt_mask.copy()
        G[G != id] = 0
        G[G == id] = 1

        D = final_pred_mask.copy()
        D[D != id] = 0
        D[D == id] = 1
        D = D.astype(float)

        # Calculate Euclidean distance for each pixel to the closest FG pixel and closest pixel
        # (logical not because the function calculates distance to BG pixels)
        dist, dist_idx = distance_transform_edt(np.logical_not(G), return_indices=True)

        E = np.abs(G - D)

        # Pixel dependency
        Et = E.copy()

        # Replace in Et where G is 0 by the value in Et[idxt] where G is 0
        x = dist_idx[0][G == 0]
        y = dist_idx[1][G == 0]
        Et[G == 0] = Et[x, y]

        # calculate truncate is necessary if we want to fix the kernel size
        sigma, window_size = 5, 7
        t = (((window_size - 1) / 2) - 0.5) / sigma
        EA = gaussian_filter(Et, sigma, truncate=t)
        min_E_EA = E.copy()
        min_E_EA[np.logical_and(G == 1, EA < E)] = EA[np.logical_and(G == 1, EA < E)]

        # Pixel importance
        B = np.ones(final_gt_mask.shape)
        B[G != 1] = 2 - np.exp((np.log(0.5) / 5.0) * dist[G != 1])

        Ew = min_E_EA * B

        tp = np.sum((1 - Ew) * G)
        fp = np.sum(Ew * (1 - G))
        r = 1 - np.mean(Ew[G == 1])
        p = tp / (tp + fp + eps)
        q = 2 * r * p / (r + p + eps)
        stats[id]["q"].append(q)
    return stats


def calculate_ap(recall, precision):
    """
        Calculates ap (average precision) measurement.
    :param recall: recall values for a concrete class
    :param precision: precision values for a concrete class
    """
    # ap = 0
    # for r in np.arange(0, 1.1, 0.1):
    #     prec_rec = precision[recall >= r]
    #     if len(prec_rec) > 0:
    #         ap += np.amax(prec_rec)
    # # By definition AP = sum(max(precision whose recall is above r))/11
    # ap /= 11

    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calculate_mAP(stats):
    """
        Calculates and shows mAP measurement for every class and mean for all classes.
    :param stats: json with statistics for all classes
    """
    aps = []
    for label in stats:
        label_stats = stats[label]
        tp = np.array(label_stats["tp"])
        fp = np.array(label_stats["fp"])
        # scores = np.array(label_stats["scores"])
        scores = np.around(np.array(label_stats["scores"]), 3)
        ids = np.argsort(-scores)
        total = label_stats["total"]
        accumulated_tp = np.cumsum(tp[ids])
        accumulated_fp = np.cumsum(fp[ids])
        recall = accumulated_tp / total
        precision = accumulated_tp / (accumulated_fp + accumulated_tp)
        ap = calculate_ap(recall, precision)
        aps.append(ap)
        print('AP for {} = {:.4f}'.format(label_stats["label"], ap))
    print("mAP: {}".format(float(np.mean(aps))))


def calculate_f_w_beta_measurement(stats):
    """
        Calculates and shows FwB measurement for every class and mean for all classes.
    :param stats: json with statistics for all classes
    """
    f_w_beta = []
    for label in stats:
        label_stats = stats[label]
        fwb = np.mean(label_stats["q"])
        f_w_beta.append(fwb)
        print('Fwb for {} = {:.4f}'.format(label_stats["label"], fwb))
    print('Fwb:', np.mean(f_w_beta))
