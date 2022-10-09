import torch
from torchvision.models.vision_transformer import VisionTransformer


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
