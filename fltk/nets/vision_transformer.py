import torch
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.models.swin_transformer import SwinTransformer


class ViT_MNIST(VisionTransformer):
    def __init__(self):
        # params taken from https://towardsdatascience.com/a-demonstration-of-using-vision-transformers-in-pytorch-mnist-handwritten-digit-recognition-407eafbc15b0
        super(ViT_MNIST, self).__init__(image_size=28,
                                        patch_size=7,
                                        num_layers=6,
                                        num_heads=8,
                                        hidden_dim=64,
                                        num_classes=10,
                                        mlp_dim=128)


class ViT_CIFAR(VisionTransformer):
    # eg ViT-Base-8
    def __init__(self):
        super(ViT_CIFAR, self).__init__(image_size=32,
                                        patch_size=8,
                                        num_layers=12,
                                        num_heads=12,
                                        hidden_dim=768,
                                        mlp_dim=3072,
                                        num_classes=100)


class ViT_Flowers(VisionTransformer):
    # eg ViT-Base-16
    def __init__(self):
        super(ViT_Flowers, self).__init__(image_size=64,
                                          patch_size=16,
                                          num_layers=12,
                                          num_heads=12,
                                          hidden_dim=768,
                                          mlp_dim=3072,
                                          num_classes=102)


class Swin_MNIST(SwinTransformer):
    # eg. Swin-T
    def __init__(self):
        super(Swin_MNIST, self).__init__(patch_size=[4, 4],
                                         embed_dim=96,
                                         depths=[2, 2, 6, 2],
                                         num_heads=[3, 6, 12, 24],
                                         window_size=[7, 7],
                                         stochastic_depth_prob=0.2,
                                         num_classes=10)


class Swin_CIFAR(SwinTransformer):
    # eg. Swin-S
    def __init__(self):
        super(Swin_CIFAR, self).__init__(patch_size=[4, 4],
                                         embed_dim=96,
                                         depths=[2, 2, 18, 2],
                                         num_heads=[3, 6, 12, 24],
                                         window_size=[7, 7],
                                         stochastic_depth_prob=0.3,
                                         num_classes=100)


class Swin_Flowers(SwinTransformer):
    # eg. Swin-S
    def __init__(self):
        super(Swin_Flowers, self).__init__(patch_size=[4, 4],
                                           embed_dim=96,
                                           depths=[2, 2, 18, 2],
                                           num_heads=[3, 6, 12, 24],
                                           window_size=[7, 7],
                                           stochastic_depth_prob=0.3,
                                           num_classes=102)
