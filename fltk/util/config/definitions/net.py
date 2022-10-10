from typing import Optional

from aenum import unique, Enum

@unique
class Nets(Enum):
    cifar100_resnet = 'Cifar100ResNet'
    cifar100_vgg = 'Cifar100VGG'
    cifar10_cnn = 'Cifar10CNN'
    cifar10_resnet = 'Cifar10ResNet'
    fashion_mnist_cnn = 'FashionMNISTCNN'
    fashion_mnist_resnet = 'FashionMNISTResNet'
    mnist_cnn = 'MNISTCNN'
    lenet_5 = 'Lenet5'
    vit_mnist = 'ViTMNIST'
    vit_cifar = 'ViTCIFAR'
    vit_flowers = 'ViTFlowers'
    swin_mnist = 'SwinMNIST'
    swin_cifar = 'SwinCIFAR'
    swin_flowers = 'SwinFlowers'

    @classmethod
    def _missing_name_(cls, name: str) -> 'Dataset':
        '''Helper function in case name could not be looked up (to support older configurations).

        Args:
            name (str): Name of Type to be looked up.

        Returns:
            Dataset: Corresponding Enum instance, if name is recognized from lower case.

        '''
        for member in cls:
            if member.name.lower() == name.lower():
                return member