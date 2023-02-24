"""
    Based on the implementation of Ross Wightman, modified by Alex Olar
    and integrated it with Pytorch Lightning
"""

import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.regression import mse
import wandb

from timm import create_model
from .unet import UnetDecoder
from .conditional_instance_norm_pp import ConditionalInstanceNormPP

class Unet(pl.LightningModule):
    """Unet is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
        num_classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        center: if ``True`` add ``Conv2dReLU`` block on encoder head

    NOTE: This is based off an old version of Unet in https://github.com/qubvel/segmentation_models.pytorch
    """

    def __init__(
            self,
            backbone='resnet50',
            backbone_kwargs=None,
            backbone_indices=None,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            in_chans=1,
            num_classes=1,
            center=False,
            norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        # NOTE some models need different backbone indices specified based on the alignment of features
        # and some models won't have a full enough range of feature strides to work properly.
        encoder = create_model(
            backbone, features_only=True, out_indices=backbone_indices, in_chans=in_chans,
            pretrained=True, **backbone_kwargs)
        encoder_channels = encoder.feature_info.channels()[::-1]
        self.encoder = encoder

        if not decoder_use_batchnorm:
            norm_layer = None
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=num_classes,
            norm_layer=norm_layer,
            center=center,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self.encoder(x)
        x.reverse()  # torchscript doesn't work with [::-1]
        x = self.decoder(x, y)
        return x

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['std_idx']
        stds = batch['std']
        noisy_x = batch['noisy_image']

        score = self.forward(noisy_x, y)
        out = score + ( noisy_x - x ) / torch.square(stds.view(-1, 1, 1, 1))

        out = torch.sum(torch.square( out ), dim=(1, 2, 3), keepdim=False) # [bs]
        loss = .5 * torch.mean(torch.square(stds.view(-1)) * out)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['std_idx']
        stds = batch['std']
        noisy_x = batch['noisy_image']

        score = self.forward(noisy_x, y)
        out = score + ( noisy_x - x ) / torch.square(stds.view(-1, 1, 1, 1))

        out = torch.sum(torch.square( out ), dim=(1, 2, 3), keepdim=False) # [bs]
        loss = .5 * torch.mean(torch.square(stds.view(-1)) * out)

        self.log('valid_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=1e-4)