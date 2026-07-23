# segmentation/seg_model.py
#
# U-Net model for Hard Exudate segmentation, built via
# segmentation_models_pytorch (smp). Uses a pretrained ResNet34 encoder —
# with only 54 training images, transfer learning from ImageNet matters a
# lot more here than for the Phase 1 classifier's much larger dataset.

import segmentation_models_pytorch as smp


def build_unet():
    """
    U-Net, ResNet34 encoder pretrained on ImageNet, single-channel output
    (binary hard-exudate mask). No activation — sigmoid is applied inside
    the loss (BCEWithLogitsLoss/DiceLoss) or explicitly at inference time,
    not baked into the model, so raw logits stay numerically stable.
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = build_unet()
    total, trainable = count_parameters(model)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
