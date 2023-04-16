import tensorflow as tf
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from utils import bbox_utils
import numpy as np
import cv2

background = [200, 222, 250, 0]
contain = [255, 0, 0, 100]
cut = [0, 153, 0, 100]
display = [192, 192, 192, 100]
engine = [96, 96, 96, 100]
grasp = [0, 102, 204, 100]
hit = [102, 0, 102, 200]
pound = [255, 204, 229, 200]
support = [102, 51, 0, 200]
w_grasp = [255, 255, 51, 100]
scoop = [255, 120, 0, 100]
label_colors_iit = np.array([background, contain, cut, display, engine, grasp, hit, pound, support, w_grasp])
label_colors_umd = np.array([background, grasp, cut, scoop, contain, pound, support, w_grasp])


def draw_grid_map(img, grid_map, stride):
    """Drawing grid intersection on given image.
    inputs:
        img = (height, width, channels)
        grid_map = (output_height * output_width, [y_index, x_index, y_index, x_index])
            tiled x, y coordinates
        stride = number of stride
    outputs:
        array = (height, width, channels)
    """
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    counter = 0
    for grid in grid_map:
        draw.rectangle((
            grid[0] + stride // 2 - 2,
            grid[1] + stride // 2 - 2,
            grid[2] + stride // 2 + 2,
            grid[3] + stride // 2 + 2), fill=(255, 255, 255, 0))
        counter += 1
    plt.figure()
    plt.imshow(image)
    plt.show()

def draw_bboxes(imgs, bboxes):
    """Drawing bounding boxes on given images.
    inputs:
        imgs = (batch_size, height, width, channels)
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
            in normalized form [0, 1]
    """
    colors = tf.constant([[1, 0, 0, 1]], dtype=tf.float32)
    imgs_with_bb = tf.image.draw_bounding_boxes(imgs, bboxes, colors)
    plt.figure()
    for img_with_bb in imgs_with_bb:
        plt.imshow(img_with_bb)
        plt.show()

def draw_bboxes_with_labels(img, true_bboxes, true_labels, bboxes, label_indices, probs, labels,
                            use_masks=False, true_masks=None, mask_ids=None, pred_masks=None):
    """Drawing bounding boxes with labels on given image.
    inputs:
        img = (height, width, channels)
        bboxes = (total_bboxes, [y1, x1, y2, x2])
            in denormalized form
        label_indices = (total_bboxes)
        probs = (total_bboxes)
        labels = [labels string list]
    """
    colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)
    image = tf.keras.preprocessing.image.array_to_img(img)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    plt.rcParams['figure.figsize'] = [10, 10]

    # add overlay for masks
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    drawing = ImageDraw.Draw(overlay)

    draw_true_bboxes(true_bboxes, true_labels, labels, draw)

    for index, bbox in enumerate(bboxes):
        y1, x1, y2, x2 = tf.split(bbox, 4)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        label_index = int(label_indices[index])
        color = tuple(colors[label_index].numpy())
        label_text = "{0} {1:0.3f}".format(labels[label_index], probs[index])
        print(labels[label_index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

    if use_masks:
        # draw predicted masks
        image = draw_true_masks(pred_masks, colors, image, drawing, overlay, labels=probs.numpy().astype(int))
        # image = draw_pred_masks(pred_masks, colors, image, drawing, overlay, labels=probs.numpy().astype(int))

    plt.figure()
    plt.imshow(image)
    plt.show()

def reset_mask_ids(mask, before_uni_ids):
    # reset ID mask values from [0, 1, 4] to [0, 1, 2] to resize later
    counter = 0
    for id in before_uni_ids:
        mask[mask == id] = counter
        counter += 1
    return mask

def convert_mask_to_original_ids_manual(mask, original_uni_ids):
    # TODO: speed up!!!
    good_range = 0.005
    temp_mask = np.copy(mask)  # create temp mask to do np.around()
    temp_mask = np.around(temp_mask, decimals=0)  # round 1.6 -> 2., 1.1 -> 1.
    current_uni_ids = np.unique(temp_mask)

    out_mask = np.full(mask.shape, 0, 'float32')

    mh, mw = mask.shape
    for i in range(mh - 1):
        for j in range(mw - 1):
            for k in range(1, len(current_uni_ids)):
                if mask[i][j] > (current_uni_ids[k] - good_range) and mask[i][j] < (current_uni_ids[k] + good_range):
                    out_mask[i][j] = original_uni_ids[k]
                    # mask[i][j] = current_uni_ids[k]
    #     const = 0.005
    #     out_mask = original_uni_ids[(np.abs(mask - original_uni_ids[:,None,None]) < const).argmax(0)]
    return out_mask

def draw_bboxes_with_labels_and_masks(img, bboxes, label_indices, probs, labels,
                            use_masks=False, pred_masks=None, aff_labels=None, dataset='iit'):
    """Drawing bounding boxes with labels on given image.
    inputs:
        img = (height, width, channels)
        bboxes = (total_bboxes, [y1, x1, y2, x2])
            in denormalized form
        label_indices = (total_bboxes)
        probs = (total_bboxes)
        labels = [labels string list]
    """
    # colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)
    image = tf.keras.preprocessing.image.array_to_img(img)
    image_copy = image.copy()
    img_width, img_height = image.size
    plt.rcParams['figure.figsize'] = [10, 10]

    curr_mask = np.full((img_height, img_width), 0.0, 'float')

    for index, bbox in enumerate(bboxes):
        draw = ImageDraw.Draw(image)

        # add overlay for masks
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        drawing = ImageDraw.Draw(overlay)
        # draw_true_bboxes(true_bboxes, true_labels, labels, draw)
        # curr_mask = np.full((img_height, img_width), 0.0, 'float')

        y1, x1, y2, x2 = tf.split(bbox, 4)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        label_index = int(label_indices[index])
        # color = tuple(colors[label_index].numpy())
        color = tuple([0, 0, 0])
        label_text = "{0} {1:0.3f}".format(labels[label_index], probs[index])
        print(labels[label_index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

        # show corresponding masks
        if use_masks:
            mask = pred_masks[index]
            # Calculate max index for each position in the mask -> calculate affordance label
            mask = np.argmax(mask, axis=2)

            # calculate distinct affordances avoiding 0
            # affordance_labels = np.unique(mask)[1:]
            original_affordance_labels = np.unique(mask)
            # print(original_affordance_labels)

            # sort before_uni_ids and reset [0, 1, 7] to [0, 1, 2]
            original_affordance_labels.sort()
            print(np.take(aff_labels, original_affordance_labels))
            mask = reset_mask_ids(mask, original_affordance_labels)

            # resize mask wrt bbox size and convert to original affordance label ids
            mask = cv2.resize(mask.astype('float'), (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
            mask = convert_mask_to_original_ids_manual(mask, original_affordance_labels)

            # TODO: add assert to check that the ids have not changed
            # original_affordance_labels = np.unique(mask).astype(int)
            # print(original_affordance_labels)

            # add mask values to current mask but preserving the maximum index -> 0 less importance, 10 max importance
            x1, x2, y1, y2 = int(x1.numpy()[0]), int(x2.numpy()[0]), int(y1.numpy()[0]), int(y2.numpy()[0])
            provisional_mask = curr_mask[y1:y2, x1:x2]
            curr_mask[y1:y2, x1:x2] = np.maximum(mask, provisional_mask)  # assign to output mask

            # for aff_label in affordance_labels:
            # mask_affordance = mask.copy()
            # mask_affordance[mask_affordance != aff_label] = 0
            # mask_affordance[mask_affordance != 0] = 255
            # mask1 = Image.fromarray(mask_affordance.astype(np.uint8), mode='L')
            # print(aff_label)
            # # color = tuple(colors[aff_label])
            # color_roi_mask = np.take(label_colours, mask.astype('int32'), axis=0)
            # plt.imshow(color_roi_mask)
            # plt.show()

    if use_masks:
        curr_mask = curr_mask.astype('uint8')
        label_colours = label_colors_iit if dataset == 'iit' else label_colors_umd
        color_curr_mask = label_colours.take(curr_mask, axis=0)
        color_curr_mask = Image.fromarray(color_curr_mask.astype(np.uint8), mode='RGBA')
        image.paste(color_curr_mask, (0,0), color_curr_mask)

    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    image.save("Classified_img.png")
    image = image_copy.copy()

def draw_true_bboxes(true_bboxes, true_labels, labels, draw):
    for index, bbox in enumerate(true_bboxes):
        if bbox.shape != (4,):
            continue
        # x1, y1, x2, y2 = tf.split(bbox, 4) #for iit dataset
        y1, x1, y2, x2 = tf.split(bbox, 4) #for tf datasets
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        color = tuple([0, 0, 0])
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label_text = "{0}".format(labels[true_labels[index]])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)

def draw_predictions(img, true_bboxes, true_labels, pred_bboxes, pred_labels, pred_scores, labels, batch_size,
                     use_masks=False, true_masks=None, mask_ids=None, pred_masks=None):
    img_height, img_width = img.shape[0], img.shape[1]
    denormalized_true_bboxes = bbox_utils.denormalize_bboxes(true_bboxes, img_height, img_width)
    denormalized_bboxes = bbox_utils.denormalize_bboxes(pred_bboxes, img_height, img_width)
    draw_bboxes_with_labels(img, denormalized_true_bboxes, true_labels, denormalized_bboxes, pred_labels, pred_scores, labels,
                     use_masks, true_masks, mask_ids, pred_masks)

def draw_predictions_with_masks(img, pred_bboxes, pred_labels, pred_scores, labels, batch_size,
                     use_masks=False, pred_masks=None, aff_labels=None, dataset='iit'):
    img_height, img_width = img.shape[0], img.shape[1]
    # denormalized_true_bboxes = bbox_utils.denormalize_bboxes(true_bboxes, img_height, img_width)
    denormalized_bboxes = bbox_utils.denormalize_bboxes(pred_bboxes, img_height, img_width)
    draw_bboxes_with_labels_and_masks(img, denormalized_bboxes, pred_labels,
                                      pred_scores, labels,
                                      use_masks, pred_masks, aff_labels, dataset)

def draw_proposals(cfg, img, true_bboxes, true_labels, pred_bboxes, pred_labels,
                   pred_deltas_proposals, pred_labels_proposals, all_rois, rois_ok,
                   labels, batch_size,
                   use_masks=False, true_masks=None, mask_ids=None, pred_masks=None, aff_labels=None):
    img_height, img_width = img.shape[0], img.shape[1]
    denormalized_true_bboxes = bbox_utils.denormalize_bboxes(true_bboxes, img_height, img_width)

    # pred_deltas_proposals *= cfg.VARIANCES
    # pred_bboxes_proposals = bbox_utils.get_bboxes_from_deltas(pred_deltas_proposals, all_rois)
    # denormalized_bboxes_proposals = bbox_utils.denormalize_bboxes(pred_bboxes_proposals, img_height, img_width)
    denormalized_selected_bboxes = bbox_utils.denormalize_bboxes(rois_ok, img_height, img_width)
    draw_proposals2(cfg, img, denormalized_true_bboxes, true_labels, denormalized_selected_bboxes, labels,
                                      use_masks, true_masks, mask_ids, pred_masks, aff_labels)

def draw_proposals2(cfg, img, true_bboxes, true_labels, bboxes, labels,
                    use_masks=False, true_masks=None, mask_ids=None, pred_masks=None, aff_labels=None):
    """Drawing bounding boxes with labels on given image.
    inputs:
        img = (height, width, channels)
        bboxes = (total_bboxes, [y1, x1, y2, x2])
            in denormalized form
        label_indices = (total_bboxes)
        probs = (total_bboxes)
        labels = [labels string list]
    """
    original_image = tf.keras.preprocessing.image.array_to_img(img)
    plt.rcParams['figure.figsize'] = [6, 6]

    image = original_image.copy()

    draw = ImageDraw.Draw(original_image)
    draw_true_bboxes(true_bboxes, true_labels, labels, draw)

    # show original image
    plt.imshow(original_image)
    plt.show()

    print('number of proposals', bboxes.shape[0])

    for index, bbox in enumerate(bboxes):
        y1, x1, y2, x2 = tf.split(bbox, 4)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue

        # crop original img to bbox proposal and resize to mask size
        img = image.copy()
        x1, x2, y1, y2 = int(x1.numpy()[0]), int(x2.numpy()[0]), int(y1.numpy()[0]), int(y2.numpy()[0])
        img = img.crop((x1, y1, x2, y2))
        img = img.resize((cfg.TRAIN_MASK_SIZE, cfg.TRAIN_MASK_SIZE))

        # show corresponding masks
        if use_masks:
            # select corresponding mask
            mask = pred_masks[index]
            mask = mask.numpy()
            print('affordance labels:', np.unique(mask))
            curr_mask = mask  # assign to output mask

            # show mask
            curr_mask = curr_mask.astype('uint8')
            color_curr_mask = label_colours.take(curr_mask, axis=0)
            color_curr_mask = Image.fromarray(color_curr_mask.astype(np.uint8), mode='RGBA')
            img.paste(color_curr_mask, (0,0), color_curr_mask)

        plt.figure()
        plt.imshow(img)
        plt.show()

def draw_ground_truth(img, true_bboxes, true_labels, true_masks, labels, dataset):
    img_height, img_width = img.shape[0], img.shape[1]
    true_bboxes = bbox_utils.denormalize_bboxes(true_bboxes, img_height, img_width)

    # Calculate random colors for all possible affordances and set alpha
    # colors = np.random.randint(0, 256, (len(labels), 4), np.int)
    # colors[:, 3] = 180
    # Choose colors for affordances
    colors = label_colors_iit if dataset == 'iit' else label_colors_umd

    image = tf.keras.preprocessing.image.array_to_img(img)
    # image.putalpha(255)
    draw = ImageDraw.Draw(image)
    plt.rcParams['figure.figsize'] = [10, 10]

    # add overlay for masks
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    drawing = ImageDraw.Draw(overlay)

    draw_true_bboxes(true_bboxes, true_labels, labels, draw)

    # draw masks
    image = draw_true_masks(true_masks, colors, image, overlay)
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# draw all masks using different colors for each affordance label
def draw_true_masks(true_masks, colors, image, overlay, labels=None):
    for index, mask in enumerate(true_masks):
        mask = mask.numpy()

        affordance_labels = np.unique(mask)[1:].astype(np.int)  # calculate distinct affordances avoiding 0
        print(affordance_labels)
        # for aff_label in affordance_labels:
        mask_affordance = mask.copy()
        # mask_affordance[mask_affordance != aff_label] = 0
        # mask_affordance[mask_affordance != 0] = 255
        mask_affordance = mask_affordance.astype(np.uint8)
        color_curr_mask = colors.take(mask_affordance, axis=0)
        color_curr_mask = Image.fromarray(color_curr_mask.astype(np.uint8), mode='RGBA')
        image.paste(color_curr_mask, (0, 0), color_curr_mask)
    return image
