import logging
from typing import Type, Dict

import torch

from fltk.util.config.definitions.net import Nets
from .cifar_100_resnet import Cifar100ResNet
from .cifar_100_vgg import Cifar100VGG, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .cifar_10_cnn import Cifar10CNN
from .cifar_10_resnet import Cifar10ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .fashion_mnist_cnn import FashionMNISTCNN
from .fashion_mnist_resnet import FashionMNISTResNet
from .mnist_cnn import MNIST_CNN
from .reddit_lstm import RNNModel
from .simple import SimpleMnist, SimpleNet, Lenet5

# Add custom models
from .vision_transformer import ViT_MNIST, ViT_Flowers, ViT_CIFAR, Swin_MNIST, Swin_CIFAR, Swin_Flowers
from .convnext import ConvNext_MNIST, ConvNext_CIFAR, ConvNext_Flowers
from .efficientnetv2 import EfficientNetV2_MNIST, EfficientNetV2_CIFAR, EfficientNetV2_Flowers


def _available_nets() -> Dict[Nets, Type[torch.nn.Module]]:
    """
    Function to acquire networks provided by the nets package.
    @return: Dictionary containing mapping from `Nets` definitions to Typed implementation.
    @rtype: Dict[Nets, Type[torch.nn.Module]]
    """
    return {
        Nets.cifar100_resnet: Cifar100ResNet,
        Nets.cifar100_vgg: Cifar100VGG,
        Nets.cifar10_cnn: Cifar10CNN,
        Nets.cifar10_resnet: Cifar10ResNet,
        Nets.fashion_mnist_cnn: FashionMNISTCNN,
        Nets.fashion_mnist_resnet: FashionMNISTResNet,
        Nets.mnist_cnn: MNIST_CNN,
        Nets.lenet_5: Lenet5,
        Nets.vit_mnist: ViT_MNIST,
        Nets.vit_cifar: ViT_CIFAR,
        Nets.vit_flowers: ViT_Flowers,
        Nets.swin_mnist: Swin_MNIST,
        Nets.swin_cifar: Swin_CIFAR,
        Nets.swin_flowers: Swin_Flowers,
        Nets.convnext_mnist: ConvNext_MNIST,
        Nets.convnext_cifar: ConvNext_CIFAR,
        Nets.convnext_flowers: ConvNext_Flowers,
        Nets.efficientnetv2_mnist: EfficientNetV2_MNIST,
        Nets.efficientnetv2_cifar: EfficientNetV2_CIFAR,
        Nets.efficientnetv2_flowers: EfficientNetV2_Flowers
    }


def get_net(name: Nets) -> Type[torch.nn.Module]:
    """
    Helper function to get specific Net implementation.
    @param name: Network definition to obtain.
    @type name: Nets
    @return: Class reference to required Network.
    @rtype: Type[torch.nn.Module]
    """
    logging.info(f"Getting net: {name}")
    return _available_nets()[name]


def get_net_split_point(name: Nets) -> int:
    """
    @deprecated Function to get split points in a network.
    @param name: Network definition to get split position/module index.
    @type name: Nets
    @return: Index of network split position.
    @rtype: int
    """
    nets_split_point = {
        Nets.cifar100_resnet: 48,
        Nets.cifar100_vgg: 28,
        Nets.cifar10_cnn: 15,
        Nets.cifar10_resnet: 39,
        Nets.fashion_mnist_cnn: 7,
        Nets.fashion_mnist_resnet: 7,
        Nets.mnist_cnn: 2,
    }
    return nets_split_point[name]
