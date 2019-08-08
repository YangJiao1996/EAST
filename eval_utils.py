import os
import cv2
import numpy as np 
import lanms
from matplotlib import pyplot as plt
from icdar import restore_rectangle
import matplotlib.patches as patches

def get_images(test_path):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG', 'bmp', 'BMP']
    for parent, dirnames, filenames in os.walk(test_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def show_pairs(image_output, image_gt, save_name):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image_output, cmap='gray')
    axes[1].imshow(image_gt, cmap='gray')
    plt.savefig(save_name)
    plt.close()

def show_single(image, save_name):
    plt.imshow(image, cmap='gray')
    plt.savefig(save_name)
    plt.close()

def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    if len(im.shape) == 3:
        h, w, _ = im.shape
    else:
        h, w = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, score_map_thresh=0.75, box_thresh=0.075, nms_thres=0.05):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    # nms part
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

    if boxes.shape[0] == 0:
        return None

    # here we filter some low score boxes by the average score map, which is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def show_polygons(img, boxes, save_name):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.imshow(img, cmap='gray')
    for box in boxes:
        poly = patches.Polygon(box, linewidth=3, edgecolor='y', facecolor='none')
        ax.add_patch(poly)
    plt.savefig(save_name)
    plt.close()

def show_cropped(img, boxes, save_name):
    for i, box in enumerate(boxes):
        my_dpi = 300
        rect = cv2.minAreaRect(box)
        im_crop = crop_rect(img, rect)
        rect_height, rect_width, _ = im_crop.shape
        fig = plt.figure(figsize=(rect_width/my_dpi, rect_height/my_dpi), frameon=False, dpi=my_dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(im_crop, cmap='gray')
        save_name = save_name + f"_box_{i}.png"
        plt.savefig(save_name)
        plt.close()

def crop_rect(img, rect):
    # Taken from https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
    # get the parameter of the small rectangle
    box = cv2.boxPoints(rect)
    rect_width = int(rect[1][0])
    rect_height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, rect_height-1],
                        [0, 0],
                        [rect_width-1, 0],
                        [rect_width-1, rect_height-1]], dtype="float32")


    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # rotate the original image
    img_crop = cv2.warpPerspective(img, M, (rect_width, rect_height))

    return img_crop

def intersection_over_union(text_box_gt, text_box, im_width, im_height, threshold=0.75):
    """ Find out whether the IoU of two bounding boxes is larger than the threshold.
    
    Arguments:
        text_box_gt {4x2 np.array} -- Ground truth bounding box
        text_box {4x2 np.array} -- Detected box
    
    Keyword Arguments:
        threshold {float} -- The IoU threshold for "matched" region. (default: {0.75})
    """


    # A "canvas" for calculating the union
    union_canvas = np.zeros((im_height, im_width))
    cv2.fillPoly(union_canvas, text_box, 1)
    cv2.fillPoly(union_canvas, text_box_gt, 1)
    union_area = np.sum(union_canvas, dtype=np.int32)
    # Two canvases for calculating the intersection
    test_intersect_canvas = np.zeros((im_height, im_width))
    gt_intersect_canvas = np.zeros((im_height, im_width))
    cv2.fillPoly(test_intersect_canvas, text_box, 1)
    cv2.fillPoly(gt_intersect_canvas, text_box_gt, 1)
    intersect_canvas = np.logical_and(test_intersect_canvas, gt_intersect_canvas, dtype=np.bool)
    intersect_area = np.sum(intersect_canvas, dtype=np.int32)

    iou = intersect_area / union_area

    if iou >= threshold:
        return True
    else:
        return False