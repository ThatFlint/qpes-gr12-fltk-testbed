from torchvision.models.efficientnet import EfficientNet, FusedMBConvConfig, MBConvConfig
from functools import partial
from torch import nn


class EfficientNetV2_MNIST(EfficientNet):
    # Eg. efficientnet_v2_s
    def __init__(self):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
        super(EfficientNetV2_MNIST, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                                   dropout=0.2,
                                                   last_channel=last_channel,
                                                   norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
                                                   num_classes=10)


class EfficientNetV2_CIFAR(EfficientNet):
    # Eg. efficientnet_v2_s
    def __init__(self):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
        super(EfficientNetV2_CIFAR, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                                   dropout=0.2,
                                                   last_channel=last_channel,
                                                   norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
                                                   num_classes=100)


class EfficientNetV2_Flowers(EfficientNet):
    # Eg. efficientnet_v2_s
    def __init__(self):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
        super(EfficientNetV2_Flowers, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                                   dropout=0.2,
                                                   last_channel=last_channel,
                                                   norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
                                                   num_classes=102)
