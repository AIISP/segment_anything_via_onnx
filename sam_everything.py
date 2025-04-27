import argparse
import json
import time
from random import randint

import cv2
import numpy as np
import onnxruntime
from scipy.ndimage import zoom

retina_masks = True
conf = 0.25
iou = 0.7
agnostic_nms = False


def clip_coords(coords, shape):
    """
    Clip line coordinates to the image boundaries.

    Args:
        coords (torch.Tensor | numpy.ndarray): A list of line coordinates.
        shape (tuple): A tuple of integers representing the size of the image in the format (height, width).

    Returns:
        (None): The function modifies the input `coordinates` in place, by clipping each coordinate to the image boundaries.
    """

    coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
    coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False):
    """
    Rescale segment coordinates (xyxy) from img1_shape to img0_shape

    Args:
      img1_shape (tuple): The shape of the image that the coords are from.
      coords (torch.Tensor): the coords to be scaled
      img0_shape (tuple): the shape of the image that the segmentation is being applied to
      ratio_pad (tuple): the ratio of the image size to the padded image size.
      normalize (bool): If True, the coordinates will be normalized to the range [0, 1]. Defaults to False

    Returns:
      coords (torch.Tensor): the segmented image.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[..., 0] -= pad[0]  # x padding
    coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords


def masks2segments(masks, strategy='largest'):
    """
    It takes a list of masks(n,h,w) and returns a list of segments(n,xy)

    Args:
      masks (torch.Tensor): the output of the model, which is a tensor of shape (batch_size, 160, 160)
      strategy (str): 'concat' or 'largest'. Defaults to largest

    Returns:
      segments (List): list of segment masks
    """
    segments = []
    for x in masks.int().cpu().numpy().astype('uint8'):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == 'concat':  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == 'largest':  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype('float32'))
    return segments


class SimpleClass:
    """
    Ultralytics SimpleClass is a base class providing helpful string representation, error reporting, and attribute
    access methods for easier debugging and usage.
    """

    def __str__(self):
        """Return a human-readable string representation of the object."""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith('_'):
                if isinstance(v, SimpleClass):
                    # Display only the module and class name for subclasses
                    s = f'{a}: {v.__module__}.{v.__class__.__name__} object'
                else:
                    s = f'{a}: {repr(v)}'
                attr.append(s)
        return f'{self.__module__}.{self.__class__.__name__} object with attributes:\n\n' + '\n'.join(attr)

    def __repr__(self):
        """Return a machine-readable string representation of the object."""
        return self.__str__()

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class BaseTensor(SimpleClass):
    """
    Base tensor class with additional methods for easy manipulation and device handling.
    """

    def __init__(self, data, orig_shape) -> None:
        """Initialize BaseTensor with data and original shape.

        Args:
            data (torch.Tensor | np.ndarray): Predictions, such as bboxes, masks and keypoints.
            orig_shape (tuple): Original shape of image.
        """
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self):
        """Return the shape of the data tensor."""
        return self.data.shape

    def cpu(self):
        """Return a copy of the tensor on CPU memory."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        """Return a copy of the tensor as a numpy array."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def __len__(self):  # override len(results)
        """Return the length of the data tensor."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a BaseTensor with the specified index of the data tensor."""
        return self.__class__(self.data[idx], self.orig_shape)


class Masks(BaseTensor):
    """
    A class for storing and manipulating detection masks.

    Args:
        masks (torch.Tensor | np.ndarray): A tensor containing the detection masks, with shape (num_masks, height, width).
        orig_shape (tuple): Original image size, in the format (height, width).

    Attributes:
        masks (torch.Tensor | np.ndarray): A tensor containing the detection masks, with shape (num_masks, height, width).
        orig_shape (tuple): Original image size, in the format (height, width).

    Properties:
        xy (list): A list of segments (pixels) which includes x, y segments of each detection.
        xyn (list): A list of segments (normalized) which includes x, y segments of each detection.

    Methods:
        cpu(): Returns a copy of the masks tensor on CPU memory.
        numpy(): Returns a copy of the masks tensor as a numpy array.
        cuda(): Returns a copy of the masks tensor on GPU memory.
        to(): Returns a copy of the masks tensor with the specified device and dtype.
    """

    def __init__(self, masks, orig_shape) -> None:
        """Initialize the Masks class."""
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @property
    def segments(self):
        """Return segments (deprecated; normalized)."""
        return self.xyn

    @property
    def xyn(self):
        """Return segments (normalized)."""
        return [
            scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in masks2segments(self.data)]

    @property
    def xy(self):
        """Return segments (pixels)."""
        return [
            scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in masks2segments(self.data)]

    @property
    def masks(self):
        """Return the raw masks tensor (deprecated)."""
        return self.data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def crop_mask(masks, bboxes):
    """
    Crops the masks based on bounding boxes.

    Args:
      masks (np.ndarray): [n, h, w], n is number of masks
      bboxes (np.ndarray): [n, 4], bounding boxes for each mask (x1, y1, x2, y2)

    Returns:
      cropped_masks (np.ndarray): [n, h, w], cropped masks
    """
    n, h, w = masks.shape
    cropped_masks = np.zeros_like(masks)
    for i in range(n):
        x1, y1, x2, y2 = bboxes[i].astype(int)
        cropped_masks[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]
    return cropped_masks


def process_mask_native(protos, masks_in, bboxes, shape):
    """
    It takes the output of the mask head, and crops it after upsampling to the bounding boxes.

    Args:
      protos (np.ndarray): [mask_dim, mask_h, mask_w]
      masks_in (np.ndarray): [n, mask_dim], n is number of masks after nms
      bboxes (np.ndarray): [n, 4], n is number of masks after nms
      shape (tuple): the size of the input image (h,w)

    Returns:
      masks (np.ndarray): The returned masks with dimensions [h, w, n]
    """
    c, mh, mw = protos.shape  # CHW
    masks = sigmoid(np.dot(masks_in, protos.reshape(c, -1))).reshape(-1, mh, mw)

    gain = min(mh / shape[0], mw / shape[1])  # gain = old / new
    pad = ((mw - shape[1] * gain) / 2, (mh - shape[0] * gain) / 2)  # wh padding
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(mh - pad[1]), int(mw - pad[0])
    masks = masks[:, top:bottom, left:right]

    # Rescale masks to the target shape using bilinear interpolation
    zoom_factors = (1, shape[0] / masks.shape[1], shape[1] / masks.shape[2])
    masks = zoom(masks, zoom_factors, order=1)  # order=1 means bilinear interpolation

    # Crop masks based on bounding boxes
    masks = crop_mask(masks, bboxes)  # CHW
    return masks > 0.5


def overlay(image, mask, color, alpha, resize=None):
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def xywh2xyxy(box):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    converted_box = np.zeros_like(box)
    converted_box[:, 0] = box[:, 0] - box[:, 2] / 2  # x1
    converted_box[:, 1] = box[:, 1] - box[:, 3] / 2  # y1
    converted_box[:, 2] = box[:, 0] + box[:, 2] / 2  # x2
    converted_box[:, 3] = box[:, 1] + box[:, 3] / 2  # y2
    return converted_box


def nms_numpy(boxes, scores, iou_thres):
    """Perform Non-Maximum Suppression (NMS) using NumPy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thres]

    return np.array(keep)


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes using NumPy.

    Arguments:
        prediction (np.ndarray): A NumPy array of shape (batch_size, num_boxes, num_classes + 5)
            containing the predicted boxes, classes, and optionally masks.
        conf_thres, iou_thres, classes, agnostic, multi_label, labels, max_det, nc,
        max_time_img, max_nms, max_wh: Same as original PyTorch function.

    Returns:
        List[np.ndarray]: A list of length batch_size, where each element is a NumPy array with shape
            (num_boxes, 6 + num_masks) containing the kept boxes.
    """
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    batch_size = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = np.amax(prediction[:, 4:mi, :], axis=1) > conf_thres  # candidates

    time_limit = 0.5 + max_time_img * batch_size
    t = time.time()
    output = [np.zeros((0, 6 + nm))] * batch_size

    for xi, x in enumerate(prediction):
        x = x.T
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            lb = np.array(labels[xi])
            v = np.zeros((len(lb), nc + nm + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].astype(int) + 4] = 1.0  # cls
            x = np.concatenate((x, v), axis=0)

        if not x.shape[0]:
            continue

        assert x.shape[1] == 4 + nc + nm, "The second dimension size must match the split sizes."

        # 使用切片操作进行拆分
        box = x[:, :4]  # 第一部分，取前 4 列
        cls = x[:, 4:4 + nc]  # 第二部分，从第 4 列开始，取 nc 列
        mask = x[:, 4 + nc:]  # 第三部分，从第 4 + nc 列开始，取剩余的列

        box = xywh2xyxy(box)  # convert box format

        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None].astype(float), mask[i]), axis=1)
        else:
            conf = cls.max(axis=1, keepdims=True)
            j = cls.argmax(axis=1, keepdims=True)
            x = np.concatenate((box, conf, j.astype(float), mask), axis=1)[conf.ravel() > conf_thres]

        if classes is not None:
            x = x[np.isin(x[:, 5], classes)]

        n = x.shape[0]
        if not n:
            continue

        x = x[np.argsort(-x[:, 4])[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        keep = nms_numpy(boxes, scores, iou_thres)
        keep = keep[:max_det]
        if len(keep):
            output[xi] = x[keep]

        if (time.time() - t) > time_limit:
            print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break

    return output


def xywh2xyxy_numpy(x):
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)"""
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms_numpy(boxes, scores, iou_threshold):
    """Apply non-maximum suppression using NumPy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:][iou <= iou_threshold]

    return np.array(keep, dtype=np.int32)


def clip_boxes(boxes, shape):
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
      img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
      boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
      img0_shape (tuple): the shape of the target image, in the format of (height, width).
      ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                         calculated based on the size difference between the two images.

    Returns:
      boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def postprocess(preds, img, orig_imgs, retina_masks, conf, iou, agnostic_nms=False):
    """
     Post-processing function.

     Args:
         preds: Model predictions, including bounding boxes and raw segmentation masks.
         inp: Preprocessed input image tensor.
         img: Original input image (used to map results back to original dimensions).
         retina_masks: Boolean indicating whether to use Retina Mask format.
         conf: Confidence threshold for filtering low-confidence predictions.
         iou: IOU threshold for Non-Maximum Suppression (NMS).
         agnostic_nms: Boolean indicating whether to use class-agnostic NMS.
     """

    p = non_max_suppression(preds[0],
                            conf,
                            iou,
                            agnostic_nms,
                            max_det=100,
                            nc=1)
    results = []
    proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
    for i, pred in enumerate(p):
        orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        # path = self.batch[0]
        img_path = "ok"
        # if not len(pred):  # save empty boxes
        #     results.append(Results(orig_img=orig_img, path=img_path, names="segment", boxes=pred[:, :6]))
        #     continue
        if retina_masks:
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
        orig_shape = orig_img.shape[:2]
        masks = Masks(masks, orig_shape) if masks is not None else None
        results.append(masks)
    return results


def pre_processing(img_origin, imgsz=1024):
    h, w = img_origin.shape[:2]
    if h > w:
        scale = min(imgsz / h, imgsz / w)
        inp = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        nw = int(w * scale)
        nh = int(h * scale)
        a = int((nh - nw) / 2)
        inp[: nh, a:a + nw, :] = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
    else:
        scale = min(imgsz / h, imgsz / w)
        inp = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        nw = int(w * scale)
        nh = int(h * scale)
        a = int((nw - nh) / 2)

        inp[a: a + nh, :nw, :] = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
    rgb = np.array([inp], dtype=np.float32) / 255.0
    return np.transpose(rgb, (0, 3, 1, 2))


def main():
    parser = argparse.ArgumentParser(description="ONNX model prediction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output_mask_path", type=str, required=True, help="Path to save the output mask")
    parser.add_argument("--output_json_path", type=str, required=True, help="Path to save the output mask")

    args = parser.parse_args()
    model_path = args.model_path
    output_mask_path = args.output_mask_path
    output_json_path = args.output_json_path
    image_path = args.image_path
    img = cv2.imread(image_path)
    inp = pre_processing(img)
    model = onnxruntime.InferenceSession(model_path,
                                         providers=['CPUExecutionProvider'])
    ort_inputs = {model.get_inputs()[0].name: inp}
    preds = model.run(None, ort_inputs)
    data_0 = preds[0]
    data_1 = [[preds[1], preds[2], preds[3]],
              preds[4], preds[5]]
    preds = [data_0, data_1]
    result = postprocess(preds, inp, img, retina_masks, conf, iou)
    masks = result[0].data
    image_with_masks = np.copy(img)
    label_dict = {}
    for i, mask_i in enumerate(masks):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        label_dict[str(i)] = rand_color
        image_with_masks = overlay(image_with_masks, mask_i, color=rand_color, alpha=1)
    cv2.imwrite(output_mask_path, image_with_masks)
    with open(output_json_path, 'w') as f:
        json.dump(label_dict, f)  # flatten and save


if __name__ == '__main__':
    t1 = time.time()
    mask = main()
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.2f} seconds")
