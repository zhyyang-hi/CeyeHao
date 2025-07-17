"""
Perceptual loss afapted from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
"""

import torch
import torchvision
import torch.nn.functional as F
from torchmetrics.functional.image import image_gradients
from ceyehao.models.gvtn.gvtn_loss import LossGvtn


class LossFunc:
    def __init__(self, cfg) -> None:
        self.loss_f = []
        self.model = cfg.model
        self.loss_components = cfg.loss_cfg.components

        if "rae" in self.loss_components:
            self.epsilon = cfg.loss_cfg.rae_epsilon
        if "gvtn_loss" in self.loss_components:
            if len(self.loss_components) > 1:
                raise ValueError(
                    "GVTN loss should be combined with other loss functions."
                )
            self.gvtn_loss_config = cfg.loss_cfg.gvtn_loss_config

        for lf in self.loss_components:
            self.loss_f.append(self._get_loss(lf))

        self.loss_weights = cfg.loss_cfg.weights

        assert len(self.loss_f) == len(
            self.loss_weights
        )  # loss function and loss weight should have the same length

    def __call__(self, preds, labels):
        loss = 0
        if self.model == "gvtn" and "gvtn_loss" not in self.loss_components:
            preds = preds[1]

        for func, weight in zip(self.loss_f, self.loss_weights):
            loss_component = func(preds, labels) * weight
            loss += loss_component
        return loss

    def _get_loss(self, name):
        if name == "l1":
            return F.l1_loss
        elif name == "l2":
            return F.mse_loss
        elif name == "l2_grad":
            return self.mse_grad
        elif name == "l1_grad":
            return self.l1_grad
        elif name == "perceptual":
            return VGGPerceptualLoss()
        elif name == "rae":
            return self.relative_absolute_loss
        elif name == "REE":
            return self.relative_entropy_loss
        elif name == "gvtn_loss":
            return LossGvtn(**self.gvtn_loss_config)
        else:
            raise ValueError(f"Unknown loss function: {name}")

    def mse_grad(self, preds, labels):
        outputs_grad = image_gradients(preds)
        labels_grad = image_gradients(labels)
        return F.mse_loss(outputs_grad[0], labels_grad[0]) + F.mse_loss(
            outputs_grad[1], labels_grad[1]
        )

    def l1_grad(self, preds, labels):
        outputs_grad = image_gradients(preds)
        labels_grad = image_gradients(labels)
        return F.l1_loss(outputs_grad[0], labels_grad[0]) + F.l1_loss(
            outputs_grad[1], labels_grad[1]
        )

    def relative_absolute_loss(self, preds, labels):
        rae = torch.abs((preds - labels) / (labels + self.epsilon)).mean()
        return rae

    def relative_entropy_loss(self, preds, labels):
        preds_tilde = F.softmax(preds)
        labels_tilde = F.softmax(labels)

        return F.kl_div(preds_tilde, labels_tilde)


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
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
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        input = F.pad(input, (0, 0, 0, 0, 0, 1), value=0)
        target = F.pad(target, (0, 0, 0, 0, 0, 1), value=0)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(
                input, mode="bilinear", size=(224, 224), align_corners=False
            )
            target = self.transform(
                target, mode="bilinear", size=(224, 224), align_corners=False
            )
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
