# -*- coding: utf-8 -*-
"""
BTES Tree Extraction - Model
UNet / UNet++ (segmentation_models_pytorch), resnet34 encoder,
offline encoder weight loading for air-gapped environments.

@author: saurav_kumar
"""

import torch
import torchvision.models as models
import segmentation_models_pytorch as smp

from config import CFG


def build_model():

    model_name = CFG.MODEL_NAME.lower()

    if model_name == "unet":

        model = smp.Unet(
            encoder_name=CFG.ENCODER,
            encoder_weights=CFG.ENCODER_WEIGHTS,
            in_channels=CFG.IN_CHANNELS,
            classes=CFG.CLASSES,
            activation=CFG.ACTIVATION
        )

    elif model_name == "unetpp":

        model = smp.UnetPlusPlus(
            encoder_name=CFG.ENCODER,
            encoder_weights=CFG.ENCODER_WEIGHTS,
            in_channels=CFG.IN_CHANNELS,
            classes=CFG.CLASSES,
            activation=CFG.ACTIVATION
        )

    else:

        raise ValueError(
            f"Unknown CFG.MODEL_NAME '{CFG.MODEL_NAME}'. Use 'unet' or 'unetpp'."
        )

    # =====================================================
    # LOAD LOCAL ENCODER WEIGHTS (OFFLINE)
    # =====================================================

    if CFG.LOCAL_ENCODER_WEIGHT is not None and CFG.ENCODER_WEIGHTS is None:

        print("\nLoading Local Encoder Weights...")

        encoder_weights = torch.load(
            CFG.LOCAL_ENCODER_WEIGHT,
            map_location="cpu"
        )

        resnet = models.resnet34()

        resnet.load_state_dict(encoder_weights)

        model.encoder.load_state_dict(
            resnet.state_dict(),
            strict=False
        )

        print("Local Encoder Weights Loaded")

    return model
