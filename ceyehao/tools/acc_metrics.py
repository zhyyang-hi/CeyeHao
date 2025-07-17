"""perceptual laccuracy modified from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49"""

import numpy as np
import torch
import cv2
import torchvision
from easydict import EasyDict
from tqdm import tqdm
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity


class PixelwiseAccuracy(object):
    """Pixelwise accurcy metric for evaluating transforamtion tensor and profiles."""

    def __init__(self, acc_cfg: EasyDict) -> None:

        assert (
            "op_flags" in acc_cfg.keys()
        ), "cfg should contain an flag to dictate pixelwise accuracy operation"
        self.op_dict = {
            "IoU": self.calc_IoU,
            "edge": self.edge_detect,
            "match": self.matching_rate,
            "round": self.round,
        }

        self.acc_cfg = acc_cfg
        self.operations = []
        op_flags = acc_cfg.op_flags
        for i in op_flags:
            self.operations.append(self.op_dict[i])
        if "matching_error_thresholds" in acc_cfg.keys():
            if type(self.acc_cfg.matching_error_thresholds) is not list:
                self.matching_error_thresholds = [
                    self.acc_cfg.matching_error_thresholds,
                ]
        else:
            self.matching_error_thresholds = [0]

    def __call__(self, samples: list):
        """samples: list of np.ndarray, each element is a tensor of shape (B, H, W, C)"""
        assert (
            len(samples) == 2
        ), "samples should contain a test and a reference, in the correct sequence"
        if isinstance(samples[0], torch.Tensor):
            samples[0] = samples[0].cpu().detach().numpy()
        if isinstance(samples[1], torch.Tensor):
            samples[1] = samples[1].cpu().detach().numpy()
        data = np.array(samples)  # shape to 2, B, W, H, C

        for i in self.operations:
            data = i(data)
        return data

    def calc_IoU(self, samples):
        samples = samples > 0
        union_positive = (samples[0] | samples[1]).sum(axis=(-3, -2, -1))

        intersection_positive = (samples[0] & samples[1]).sum(axis=(-3, -2, -1))
        return intersection_positive / union_positive  # shape (B,)

    def matching_rate(self, samples):
        """percentage of matched points with error below the specified threshold.
        specify the threshold of correct matching in cfg
        arugments:
        samples: np.ndarray of shape (2, B, H, W, C)
        return:
        acc: np.ndarray of shape (B, (num_thres))"""
        num_points = np.prod(samples.shape[2:])  # H*W*C
        thres = self.matching_error_thresholds
        limits = np.array(
            [np.full(samples.shape[1:], limit) for limit in thres]
        )  # num_thres, B, H, W, C
        normalized_abs_diff = np.abs((samples[0] - samples[1]) / (samples[1] + 1e-6))
        normalized_abs_diff = np.expand_dims(
            normalized_abs_diff, axis=0
        )  # 1, B, H, W, C
        matched = np.sum(
            normalized_abs_diff <= limits,
            axis=tuple(range(2, normalized_abs_diff.ndim)),
        )  # num_thres, B
        acc = matched / num_points
        # transofrm to shape (B, num_thres)
        acc = acc.transpose(1, 0)
        if acc.shape[-1] == 1:
            acc = acc.squeeze(-1)
        return acc

    def edge_detect(self, samples: list):
        for i in range(len(samples)):
            samples[i] = self.dilated_edge(samples[i])
        return samples

    def round(self, samples: list, n=0):
        """round the samples"""
        samples = np.round(samples, n)
        return samples

    @staticmethod
    def dilated_edge(img: np.ndarray, dilation=5):
        """input shape (B, H, W, C)
        find the edge of the profile image (cv2 standard, single channel)"""
        grad = np.gradient(img, axis=(-3, -2))
        grad = (np.abs(grad[0]) + np.abs(grad[1])) / 2
        # take B as channel for cv2.dilate to dialte multichannels together
        if dilation is not None and dilation > 1:
            edge = grad.astype("uint8")
            edge = np.array(
                [
                    cv2.dilate(edge[i], np.ones((dilation, dilation)))
                    for i in range(edge.shape[0])
                ]
            )
        edge = edge[..., np.newaxis]  # add channel dim
        return edge


class PerceptualAccuracy(torch.nn.Module):
    def __init__(
        self,
        acc_cfg=None,
        resize=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(PerceptualAccuracy, self).__init__()
        self.acc_cfg = acc_cfg
        self.num_block = 4
        blocks = []
        blocks.append(
            torchvision.models.vgg16(weights="IMAGENET1K_V1").features[:4].eval()
        )
        blocks.append(
            torchvision.models.vgg16(weights="IMAGENET1K_V1").features[4:9].eval()
        )
        blocks.append(
            torchvision.models.vgg16(weights="IMAGENET1K_V1").features[9:16].eval()
        )
        blocks.append(
            torchvision.models.vgg16(weights="IMAGENET1K_V1").features[16:23].eval()
        )

        for bl in blocks:
            for param in bl.parameters():
                param.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.device = device
        self.to(device)
        if acc_cfg is None:
            self.include_pix_acc = False
            self.weights = [
                4.0,
                1.0,
                0.25,
                0.125,
            ]
        else:
            self.include_pix_acc = acc_cfg.include_pix_acc
            if self.include_pix_acc:
                self.weights = acc_cfg.perceptual_weights
            else:
                self.weights = acc_cfg.perceptual_weights[1:]

        pixcfg = EasyDict()
        pixcfg.op_flags = ["match"]
        pixcfg.matching_error_thresholds = 0.0
        self.pixel_metric = PixelwiseAccuracy(pixcfg)

    def __call__(self, samples, ref_cfc=False):
        """
        Args:
            samples: np.ndarray of test and reference, shape (2, (B,) H, W, C)
            ref_cfc (bool, optional): wether apply image process for confocal image to the ref.

        """
        test, ref = samples
        if len(test.shape) == 3:
            test = np.expand_dims(test, axis=0)
        if len(ref.shape) == 3:
            ref = np.expand_dims(ref, axis=0)
        accuracy = []
        test = self.pre_img_process(test)
        ref = self.pre_img_process(ref, cfc_img=ref_cfc)
        x = self.img2tensor(test)
        y = self.img2tensor(ref)
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            acc = self.pixel_metric([x.cpu(), y.cpu()])
            accuracy.append(acc)
        accuracy = np.concatenate(accuracy, axis=-1)
        if self.include_pix_acc:
            pix_acc = self.pixel_metric([test[:, :, 1], ref[:, :, 1]])
            accuracy = np.concatenate(
                [
                    np.array(pix_acc).reshape(
                        1,
                    ),
                    accuracy,
                ]
            )
            accuracy = np.average(accuracy, weights=self.weights, axis=-1)
        else:
            accuracy = np.average(accuracy, weights=self.weights, axis=-1)

        return accuracy  # shape (B,)

    def pre_img_process(self, img: np.ndarray, out_size=(224, 224), cfc_img=False):
        """
        allowed image format:
        - (H, W, (1)): single channel image
        - (H, W, 3): 3-channel image
        - (B, H, W, ): batch of single channel images
        - (B, H, W, 3): batch of 3-channel images
        processes:
        1. unify the number of channel to 3, by adding empty channels.
        2. unify the size of the image.
        3. for confocal images (cfc_img=True), apply gaussian blur and adaptive thresholding.
        4. binarize the image value.
        """
        im_shape = img.shape
        if img.shape[-1] != 3:
            if img.shape[-1] != 1:  # ((B),H,W)
                img = img[..., np.newaxis]
            empty_chn = np.zeros(img.shape)
            img = np.concatenate([empty_chn, img, empty_chn], axis=-1)

        if img.ndim == 3:  # sinlge image
            img = np.expand_dims(img, axis=0)
        elif img.ndim == 4:
            pass
        else:
            raise ValueError(
                f"pre_img_process does not accept image of shape{im_shape}"
            )

        # resize the image
        new_img = []
        for im in img:
            new_img.append(cv2.resize(im, out_size, interpolation=cv2.INTER_LINEAR))
        img = np.stack(new_img, axis=0)

        if cfc_img:
            for idx, im in enumerate(img):
                im = cv2.GaussianBlur(im, (5, 5), 3)
                cb, cg, cr = cv2.split(im)
                cr = cv2.adaptiveThreshold(
                    cr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 131, -3
                )
                cg = cv2.adaptiveThreshold(
                    cg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 131, -3
                )
                cb = cv2.adaptiveThreshold(
                    cb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 131, -3
                )
                img[idx] = cv2.merge([cb, cg, cr])

        # normalize the image to [0, 255] and then binarize
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype(np.uint8)
        img = (img > 127).astype(np.uint8) * 255

        assert img.shape[1:] == (*(out_size), 3)
        return img

    def img2tensor(self, img):
        """nomralize the image for vgg input."""
        # from cv img to tensor
        # swap channels and rescale to [0., 1.]， normalize
        tensor = torch.tensor(img.transpose(0, 3, 1, 2)) / 255
        tensor = tensor.to(self.device)
        tensor = (tensor - self.mean) / self.std
        return tensor


def calc_iou(a, b):
    """calculate the intersection over union of two binary arrays
    Args:
        a (np.ndarray):  tensor of shape (..., H, W, C)
        b (np.ndarray):  tensor (..., H, W, C)
    Returns:
        np.ndarray: iou value of shape (...)
    """
    assert a.shape == b.shape
    axis_for_sum = tuple(range(-3, 0))
    intersection = np.logical_and(a, b).sum(axis=axis_for_sum)
    union = np.logical_or(a, b).sum(axis=axis_for_sum)
    return (intersection / union).reshape(*a.shape[:-3])


def calc_matching_rate(a, b, threshold=0):
    """calculate the matching rate of two binary arrays
    Args:
        a (np.ndarray):  tensor of shape (..., H, W, C)
        b (np.ndarray):  tensor (..., H, W, C)
        threshold (int, optional): maximum allowed relative error. Defaults to 0.
    Returns:
        np.ndarray: matching rate
    """
    assert a.shape == b.shape

    axis_for_sum = tuple(range(-3, 0))
    diff = np.abs(a - b) / (b + 1e-10)
    matched = np.sum(diff <= threshold, axis=axis_for_sum)
    if len(matched.shape) == 1:
        matched = matched[np.newaxis, ...]
    return matched / np.prod(a.shape[-3:])


def get_edge(fp_tensor: np.ndarray, horin_weight=1, vert_weight=1):
    """input shape ((B,) H, W, C).
    find the edge of the profile image (cv2 standard, single channel)"""
    if fp_tensor.ndim == 3:
        fp_tensor = fp_tensor[np.newaxis, :]
    grad = np.gradient(fp_tensor, axis=(-3, -2))
    grad = horin_weight * (np.abs(grad[0]) > 0) + vert_weight * (np.abs(grad[1]) > 0)
    return grad  # shape (B, H, W, C)


class EdgeScore:
    def __init__(self, ref_fp, horiz_weight=1, vert_weight=1):
        self.ref_score = self.edge_sum(ref_fp).squeeze()
        self.hw = horiz_weight
        self.vw = vert_weight

    def edge_sum(self, fp_tensor: np.ndarray, horiz_weight=1, vert_weight=1):
        """input shape ((B,) H, W, C). (cv2 standard, single channel)"""
        grad = get_edge(fp_tensor, horiz_weight, vert_weight)
        return grad.sum(tuple(range(1, grad.ndim)))

    def __call__(self, fp_tensor):
        sum = self.edge_sum(fp_tensor, self.hw, self.vw)
        return sum.squeeze() / self.ref_score


def corr2_squared(l1, l2):

    # Calculate the mean of each array
    avg_I1 = np.mean(l1)
    avg_I2 = np.mean(l2)

    # Calculate the differences from the mean
    diff_I1 = l1 - avg_I1
    diff_I2 = l2 - avg_I2

    # Calculate the sum of products of differences
    sum_val = np.sum(diff_I1 * diff_I2)

    # Calculate the variances
    var21 = np.sum(diff_I1**2)
    var22 = np.sum(diff_I2**2)

    # Handle zero variance cases
    zero_variance_mask = var21 * var22 == 0
    result = np.zeros_like(sum_val)

    # Calculate the squared correlation coefficient
    result[~zero_variance_mask] = (sum_val[~zero_variance_mask] ** 2) / (
        var21[~zero_variance_mask] * var22[~zero_variance_mask]
    )

    return result

def ssim(samples):
    """input shape 2, b, h, w, c, return shape (b,)"""
    pouts = samples[0]
    target = samples[1]
    return np.array(
        [structural_similarity(pout, tgt, channel_axis=-1) for pout, tgt in zip(pouts, target)]
    )
