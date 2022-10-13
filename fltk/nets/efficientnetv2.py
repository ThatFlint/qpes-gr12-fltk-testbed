from torchvision.models.efficientnet import EfficientNet, FusedMBConvConfig, MBConvConfig
from functools import partial
from torch import nn


class EfficientNetV2_MNIST(EfficientNet):
    def __init__(self):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 12, 12, 2),
            #FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 12, 24, 4),
            MBConvConfig(4, 3, 2, 24, 48, 6),
            #MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 48, 64, 9),
        ]
        last_channel = 640
        super(EfficientNetV2_MNIST, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                                   dropout=0.2,
                                                   last_channel=last_channel,
                                                   norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
                                                   num_classes=10)


class EfficientNetV2_CIFAR(EfficientNet):
    def __init__(self):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 12, 12, 2),
            # FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 12, 24, 4),
            MBConvConfig(4, 3, 2, 24, 48, 6),
            # MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 48, 64, 9),
        ]
        last_channel = 640
        super(EfficientNetV2_CIFAR, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                                   dropout=0.2,
                                                   last_channel=last_channel,
                                                   norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
                                                   num_classes=100)


class EfficientNetV2_Flowers(EfficientNet):
    def __init__(self):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 12, 12, 2),
            # FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 12, 24, 4),
            MBConvConfig(4, 3, 2, 24, 48, 6),
            # MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 48, 64, 9),
        ]
        last_channel = 640
        super(EfficientNetV2_Flowers, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                                   dropout=0.2,
                                                   last_channel=last_channel,
                                                   norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
                                                   num_classes=102)
