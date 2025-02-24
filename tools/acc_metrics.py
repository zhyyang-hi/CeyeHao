"""perceptual laccuracy modified from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
"""

import numpy as np
import torch
import cv2
import torchvision
from easydict import EasyDict
from tqdm import tqdm


class PixelwiseAccuracy(object):
    """Pixelwise accurcy metric for evaluating transforamtion tensor and profiles."""

    def __init__(self, acc_cfg: EasyDict) -> None:
        assert hasattr(
            acc_cfg, "op_flags"
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

        tqdm.write(f"pixelwise accuracy operations:{op_flags}")
        if any([op == "match" for op in op_flags]):
            tqdm.write(
                f"Matching error thresholds: {acc_cfg.matching_error_thresholds}"
            )

        for i in op_flags:
            self.operations.append(self.op_dict[i])
    def __call__(self, samples: list):
        """samples: list of np.ndarray, each element is a tensor of shape ((B,) H, W, C)"""
        assert (
            len(samples) == 2
        ), "samples should contain a test and a reference, in the correct sequence"
        if isinstance(samples[0], torch.Tensor):
            samples[0] = samples[0].cpu().detach().numpy()
        if isinstance(samples[1], torch.Tensor):
            samples[1] = samples[1].cpu().detach().numpy()
        if samples[0].ndim == 3:
            samples = [
                np.expand_dims(i, axis=0) for i in samples
            ]  # shape to B, H, W, C

        data = np.array(samples)  # shape to 2, B, H, W, C
        for i in self.operations:
            data = i(data)
        return data

    def calc_IoU(self, samples):
        samples = samples > 0
        union_positive = (samples[0] | samples[1]).sum(axis=(-3, -2, -1))

        intersection_positive = (samples[0] & samples[1]).sum(axis=(-3, -2, -1))
        return intersection_positive / union_positive

    def matching_rate(self, samples):
        """percentage of matched points with error below the specified threshold.
        specify the threshold of correct matching in cfg
        arugments:
        samples: np.ndarray of shape (2, B, H, W, C)
        return:
        acc: np.ndarray of shape (B, (num_thres))"""
        num_points = np.prod(samples.shape[2:])  # H*W*C
        if type(self.acc_cfg.matching_error_thresholds) is not list:
            thres = [
                self.acc_cfg.matching_error_thresholds,
            ]
        else:
            thres = self.acc_cfg.matching_error_thresholds
        limits = np.array(
            [np.full(samples.shape[1:], limit) for limit in thres]
        )  # num_thres, B, H, W, C
        normalized_abs_diff = np.abs((samples[0] - samples[1]) / (samples[1] + 1e-6))
        normalized_abs_diff = np.expand_dims(normalized_abs_diff, axis=0) # 1, B, H, W, C
        matched = np.sum(
            normalized_abs_diff <= limits,
            axis=tuple(range(2, normalized_abs_diff.ndim)),
        ) # num_thres, B
        acc = matched / num_points
        # transofrm to shape (B, num_thres)
        acc = acc.transpose(1, 0)
        return acc

    def edge_detect(self, samples: list):
        """current implementation is for singel channel only"""
        for i in range(len(samples)):
            samples[i] = self.find_edge(samples[i])
        return samples

    def round(self, samples: list, n=0):
        """round the samples"""
        samples = np.round(samples, n)
        return samples

    @staticmethod
    def find_edge(img: np.ndarray):
        """input shape (B, H, W, C)
        find the edge of the profile image (cv2 standard, single channel)
        Warning: current implementation only support single channel"""
        img_ = img.squeeze()  # suppress channel dimension, which should equal to 1.
        grad = np.gradient(img_, axis=(-2, -1))
        grad = (np.abs(grad[0]) + np.abs(grad[1])) > 0
        edge = grad.astype("uint8") * 255  # shape (B, H, W)
        # take B as channel for cv2.dilate to dialte multichannels together
        edge = cv2.dilate(edge.transpose(1, 2, 0), np.ones((7, 7))).transpose(2, 0, 1)
        edge = np.expand_dims(edge, axis=-1)
        return edge


class PerceptualAccuracy(torch.nn.Module):
    """perceptual accuracy metric for evaluating profiles.
    supoort only green color profiles only.
    """

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
        pixcfg.pix_acc_op_flags = ["match"]
        pixcfg.matching_error_thresholds = 0.0
        self.pixel_metric = PixelwiseAccuracy(pixcfg)

    def __call__(self, samples, ref_cfc=False):
        """input samples should be

        Args:
            samples: a list of two np.ndarray of test and reference, shape ((B,) H, W, C)
            ref_cfc (bool, optional): wether apply image process for confocal image to the ref.

        Returns:
            _type_: _description_
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
        accuracy = np.stack(accuracy, axis=1)
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
            accuracy = np.average(accuracy, weights=self.weights, axis=1)
        else:
            accuracy = np.average(accuracy, weights=self.weights, axis=1)

        return accuracy

    def pre_img_process(self, img: np.ndarray, out_size=(224, 224), cfc_img=False):
        """
        [currently only support single channel comparison, positive or negative]
        preprocess the image to compare perceptual similarity.
        allowed image format:
        - (H, W): single channel image
        - (H, W, 3): 3-channel image
        - (B, H, W): stack of single channel images
        - (B, H, W, 3): stack of 3-channel images
        processes:
        1. unify the number of channel to 3, by adding empty channels.
        2. unify the size of the image.
        3. for confocal images (cfc_img=True), apply gaussian blur and adaptive thresholding.
        4. binarize the image value.
        """
        if img.shape[-1] != 3:  # single channel image
            img = np.expand_dims(img, axis=-1)
            empty_chn = np.zeros(img.shape)
            img = np.concatenate([empty_chn, img, empty_chn], axis=-1)
        elif img.shape[-1] == 3:
            # only keep green channel
            img[:, :, [0, 2]] = 0

        if img.ndim == 3:  # sinlge image
            img = np.expand_dims(img, axis=0)
        elif img.ndim == 4:
            pass
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
