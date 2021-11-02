import os
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
from utils import bbox_utils

from utils.datasets import factory, imdb
import utils.roi_data_layer.roidb as rdl_roidb

def preprocessing(image_data, final_height, final_width, apply_augmentation=False):
    """Image resizing operation handled before batch operations.
    inputs:
        image_data = tensorflow dataset image_data
        final_height = final image height after resizing
        final_width = final image width after resizing
    outputs:
        img = (final_height, final_width, channels)
        gt_boxes = (gt_box_size, [y1, x1, y2, x2])
        gt_labels = (gt_box_size)
    """
    img = image_data["image"]
    gt_boxes = image_data["objects"]["bbox"]
    gt_labels = tf.cast(image_data["objects"]["label"] + 1, tf.int32)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (final_height, final_width))
    if apply_augmentation:
        img, gt_boxes = randomly_apply_operation(flip_horizontally, img, gt_boxes)
    return img, gt_boxes, gt_labels


def preprocessing_iit_dataset(image_data, final_height, final_width, mask_reg):
    """
        Image and mask resizing, bboxes normalization and clipping before training.
    :param image_data: tensorflow dataset image_data
    :param final_height: final image height after resizing
    :param final_width: final image width after resizing
    :param mask_reg: True if masks are used in training
    :returns:
        img: final resized image (final_height, final_width, channels)
        gt_boxes: final normalized and clipped bboxes (gt_box_size, [y1, x1, y2, x2])
        gt_labels: labels for final bboxes (gt_box_size)
        gt_masks: final resized mask (final_height, final_width)
    """
    img = image_data["image"]
    original_height, original_width = tf.shape(img)[0], tf.shape(img)[1]
    original_height = tf.cast(original_height, tf.float32)
    original_width = tf.cast(original_width, tf.float32)

    gt_boxes = image_data["objects"]["bbox"]
    gt_labels = tf.cast(image_data["objects"]["label"], tf.int32)

    # resize image and mask
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (final_height, final_width))

    # read and resize mask if masks are activated
    if mask_reg:
        gt_masks = tf.cast(image_data["objects"]["mask"], tf.int32)
        gt_seg_mask_inds = tf.cast(image_data["objects"]["seg_mask_inds"], tf.int32)
        gt_masks = tf.image.resize(gt_masks, (final_height, final_width), method='nearest')
        gt_masks = tf.squeeze(gt_masks, axis=3)  # remove last dim (only 1 value)

    # change coordinates order, clip bboxes to image and normalize bboxes
    x1, y1, x2, y2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
    gt_boxes = tf.stack([y1, x1, y2, x2], axis=1)
    gt_boxes = bbox_utils.clip_bboxes(gt_boxes, original_height, original_width)
    gt_boxes = bbox_utils.normalize_bboxes(gt_boxes, original_height, original_width)

    if mask_reg:
        return img, gt_boxes, gt_labels, gt_masks, gt_seg_mask_inds
    return img, gt_boxes, gt_labels


def preprocessing_iit_dataset_no_resize(image_data, mask_reg):
    """Image resizing operation handled before batch operations.
    inputs:
        image_data = tensorflow dataset image_data
        final_height = final image height after resizing
        final_width = final image width after resizing
    outputs:
        img = (final_height, final_width, channels)
        gt_boxes = (gt_box_size, [y1, x1, y2, x2])
        gt_labels = (gt_box_size)
    """
    img = image_data["image"]
    img_size = image_data["image_shape"]
    # im_ind = image_data['im_ind']
    original_height, original_width = tf.shape(img)[0], tf.shape(img)[1]
    original_height = tf.cast(original_height, tf.float32)
    original_width = tf.cast(original_width, tf.float32)

    gt_boxes = image_data["objects"]["bbox"]
    gt_labels = tf.cast(image_data["objects"]["label"], tf.int32)

    # resize image and mask
    img = tf.image.convert_image_dtype(img, tf.float32)

    # read and resize mask if masks are activated
    if mask_reg:
        gt_masks = tf.cast(image_data["objects"]["mask"], tf.int32)
        gt_seg_mask_inds = tf.cast(image_data["objects"]["seg_mask_inds"], tf.int32)
        gt_masks = tf.squeeze(gt_masks, axis=3)  # remove last dim (only 1 value)

    # change coordinates order, clip bboxes to image and normalize bboxes
    x1, y1, x2, y2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
    gt_boxes = tf.stack([y1, x1, y2, x2], axis=1)
    gt_boxes = bbox_utils.clip_bboxes(gt_boxes, original_height, original_width)
    gt_boxes = bbox_utils.normalize_bboxes(gt_boxes, original_height, original_width)

    if mask_reg:
        return img, img_size, gt_boxes, gt_labels, gt_masks, gt_seg_mask_inds
    return img, img_size, gt_boxes, gt_labels


def get_random_bool():
    """Generating random boolean.
    outputs:
        random boolean 0d tensor
    """
    return tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5)


def randomly_apply_operation(operation, img, gt_boxes):
    """Randomly applying given method to image and ground truth boxes.
    inputs:
        operation = callable method
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    outputs:
        modified_or_not_img = (final_height, final_width, depth)
        modified_or_not_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    return tf.cond(
        get_random_bool(),
        lambda: operation(img, gt_boxes),
        lambda: (img, gt_boxes)
    )


def flip_horizontally(img, gt_boxes):
    """Flip image horizontally and adjust the ground truth boxes.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    outputs:
        modified_img = (height, width, depth)
        modified_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    flipped_img = tf.image.flip_left_right(img)
    flipped_gt_boxes = tf.stack([gt_boxes[..., 0],
                                1.0 - gt_boxes[..., 3],
                                gt_boxes[..., 2],
                                1.0 - gt_boxes[..., 1]], -1)
    return flipped_img, flipped_gt_boxes


def get_dataset(name, split, data_dir):
    """Get tensorflow dataset split and info.
    inputs:
        name = name of the dataset, voc/2007, voc/2012, etc.
        split = data split string, should be one of ["train", "validation", "test"]
        data_dir = read/write path for tensorflow datasets
    outputs:
        dataset = tensorflow dataset split
        info = tensorflow dataset info
    """
    assert split in ["train", "train+validation", "validation", "test"]
    dataset, info = tfds.load(name, split=split, data_dir=data_dir, with_info=True)
    return dataset, info


def get_total_item_size(info, split):
    """Get total item size for given split.
    inputs:
        info = tensorflow dataset info
        split = data split string, should be one of ["train", "validation", "test"]
    outputs:
        total_item_size = number of total items
    """
    assert split in ["train", "train+validation", "validation", "test"]
    if split == "train+validation":
        return info.splits["train"].num_examples + info.splits["validation"].num_examples
    return info.splits[split].num_examples


def get_labels(info):
    """Get label names list.
    inputs:
        info = tensorflow dataset info
    outputs:
        labels = [labels list]
    """
    if info.name == 'coco':
        return info.features['objects']['label'].names
    else:
        return info.features["labels"].names


def get_custom_imgs(custom_image_path):
    """Generating a list of images for given path.
    inputs:
        custom_image_path = folder of the custom images
    outputs:
        custom image list = [path1, path2]
    """
    img_paths = []
    for path, dir, filenames in os.walk(custom_image_path):
        for filename in filenames:
            img_paths.append(os.path.join(path, filename))
        break
    return img_paths


def custom_data_generator(img_paths, final_height, final_width):
    """Yielding custom entities as dataset.
    inputs:
        img_paths = custom image paths
        final_height = final image height after resizing
        final_width = final image width after resizing
    outputs:
        img = (final_height, final_width, depth)
        dummy_gt_boxes = (None, None)
        dummy_gt_labels = (None, )
    """
    for img_path in img_paths:
        image = Image.open(img_path)
        resized_image = image.resize((final_width, final_height), Image.LANCZOS)
        img = np.array(resized_image)
        img = tf.image.convert_image_dtype(img, tf.float32)
        yield img, tf.constant([[]], dtype=tf.float32), tf.constant([], dtype=tf.int32)


def get_data_types_training():
    """Generating data types with image shape for tensorflow datasets.
    outputs:
        data types = output data types for (images, images shape, ground truth boxes, ground truth labels, mask path and mask)
    """
    return {'image': tf.string, 'image_shape': tf.float32, 'flipped': tf.bool, 'objects': {'bbox': tf.float32,
            'label': tf.int32, 'seg_mask_inds': tf.int32, 'mask_path': tf.string, 'mask': tf.float32}}


def get_data_types():
    """Generating data types with image shape for tensorflow datasets.
    outputs:
        data types = output data types for (images, images shape, ground truth boxes, ground truth labels, mask path and mask)
    """
    return {'image': tf.string, 'image_shape': tf.float32, 'objects': {'bbox': tf.float32, 'label': tf.int32,
           'seg_mask_inds': tf.int32, 'mask_path': tf.string, 'mask': tf.float32}}


def get_data_shapes(masks):
    """Generating data shapes for tensorflow datasets depending on if we need to load the masks or not
    outputs:
        data shapes with masks = output data shapes for (images, ground truth boxes, ground truth labels, masks, mask id labels)
        data shapes without masks = output data shapes for (images, ground truth boxes, ground truth labels)
    """
    if masks:
        return ([None, None, None], [None, ], [None, None], [None, ], [None, None, None], [None, ])
    else:
        return ([None, None, None], [None, ], [None, None], [None, ])


def get_padding_values(masks):
    """Generating padding values for missing values in batch for tensorflow datasets depending on if we need to load the masks
    outputs:
        padding values with masks = padding values with dtypes for (images, ground truth boxes, ground truth labels, masks, mask id labels)
        padding values without masks = padding values with dtypes for (images, ground truth boxes, ground truth labels)
    """
    if masks:
        return (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(0, tf.float32),
                tf.constant(-1, tf.int32),  tf.constant(-1, tf.int32), tf.constant(-1, tf.int32))
    else:
        return (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(0, tf.float32),
                tf.constant(-1, tf.int32))


def get_data_types_resize():
    """Generating data types with img resizing for tensorflow datasets.
    outputs:
        data types = output data types for (images, ground truth boxes, ground truth labels, mask path and mask)
    """
    return {'image': tf.string, 'objects': {'bbox': tf.float32, 'label': tf.int32, 'seg_mask_inds': tf.int32,
                                            'mask_path': tf.string, 'mask': tf.float32}}

def get_data_shapes_resize(masks):
    """Generating data shapes with img resizing for tensorflow datasets depending on if we need to load the masks or not
    outputs:
        data shapes with masks = output data shapes for (images, ground truth boxes, ground truth labels, masks, mask id labels)
        data shapes without masks = output data shapes for (images, ground truth boxes, ground truth labels)
    """
    if masks:
        return ([None, None, None], [None, None], [None, ], [None, None, None], [None, ])
    else:
        return ([None, None, None], [None, None], [None,])


def get_padding_values_resize(masks):
    """Generating padding values with img resizing for missing values in batch for tensorflow datasets depending on if we need to load the masks
    outputs:
        padding values with masks = padding values with dtypes for (images, ground truth boxes, ground truth labels, masks, mask id labels)
        padding values without masks = padding values with dtypes for (images, ground truth boxes, ground truth labels)
    """
    if masks:
        return (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32),
                tf.constant(-1, tf.int32), tf.constant(-1, tf.int32))
    else:
        return (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))


def get_training_roidb(imdb, flipped, mode):
    """
        Returns a roidb (Region of Interest database) for use in training.
    :param imdb: bboxes not normalized (batch_size, total_bboxes, [y1, x1, y2, x2])
    :param flipped: True if we want to flip all images and add them to the dataset
    :param mode: training or inference
    :returns: Region of Interest database for use in training
    """
    if mode == 'training' and flipped:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
    print('Preparing training data...')
    rdl_roidb.prepare_roidb(imdb)
    print('done')
    return imdb.roidb


def combined_roidb(imdb_names, proposal_method, flipped, mode="inference"):
    def get_roidb(imdb_name):
        imdb = factory.get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(proposal_method)
        print('Set proposal method: {:s}'.format(proposal_method))
        roidb = get_training_roidb(imdb, flipped, mode)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = imdb.imdb(imdb_names)
    else:
        imdb = factory.get_imdb(imdb_names)
    return imdb, roidb


def read_mask(mask_file):
    """
        Read mask file from png file
    :param mask_file: path to mask file
    :returns: decoded image
    """
    mask_im = tf.io.read_file(mask_file)
    mask_im = tf.io.decode_png(mask_im, channels=1)
    return mask_im
