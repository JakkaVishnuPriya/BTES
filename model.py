# -*- coding: utf-8 -*-
"""
BTES Tree Extraction - Model
UNet / UNet++ (segmentation_models_pytorch), resnet34 encoder,
offline encoder weight loading for air-gapped environments.

@author: saurav_kumar
"""
import os
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
            activation=CFG.ACTIVATION,
        )

    elif model_name == "unetpp":

        model = smp.UnetPlusPlus(
            encoder_name=CFG.ENCODER,
            encoder_weights=CFG.ENCODER_WEIGHTS,
            in_channels=CFG.IN_CHANNELS,
            classes=CFG.CLASSES,
            activation=CFG.ACTIVATION,
        )

    else:

        raise ValueError(
            f"Unknown CFG.MODEL_NAME '{CFG.MODEL_NAME}'. Use 'unet' or 'unetpp'."
        )

    # =====================================================
    # PRIORITY 1 : LOAD PRETRAINED FULL MODEL
    # =====================================================

    if (
        CFG.PRETRAIN_DIR is not None
        and os.path.exists(CFG.PRETRAIN_DIR)
    ):

        print("\n======================================")
        print("Loading Pretrained Model...")
        print(CFG.PRETRAIN_DIR)

        checkpoint = torch.load(
            CFG.PRETRAIN_DIR,
            map_location="cpu"
        )

        # if saved as checkpoint dictionary
        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:

                model.load_state_dict(
                    checkpoint["model_state_dict"],
                    strict=False
                )

            elif "state_dict" in checkpoint:

                model.load_state_dict(
                    checkpoint["state_dict"],
                    strict=False
                )

            else:

                model.load_state_dict(
                    checkpoint,
                    strict=False
                )

        else:

            model.load_state_dict(
                checkpoint,
                strict=False
            )

        print("Pretrained Model Loaded Successfully")
        print("======================================")

    # =====================================================
    # PRIORITY 2 : LOAD ONLY ENCODER WEIGHTS
    # =====================================================

    elif (
        CFG.LOCAL_ENCODER_WEIGHT is not None
        and os.path.exists(CFG.LOCAL_ENCODER_WEIGHT)
        and CFG.ENCODER_WEIGHTS is None
    ):

        print("\n======================================")
        print("Loading Local Encoder Weights...")

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

        print("Local Encoder Loaded")
        print("======================================")

    else:

        print("\nTraining From Scratch")

    return model