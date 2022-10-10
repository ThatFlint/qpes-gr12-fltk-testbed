from torchvision.models.convnext import ConvNeXt, CNBlockConfig

"""We train ConvNeXts for 300
epochs using AdamW [46] with a learning rate of 4e-3.
There is a 20-epoch linear warmup and a cosine decaying
schedule afterward. We use a batch size of 4096 and a
weight decay of 0.05. https://arxiv.org/pdf/2201.03545.pdf"""

class ConvNext_MNIST(ConvNeXt):
    # Eg. ConvNext_Tiny
    def __init__(self):
        block_setting = [
            CNBlockConfig(96, 192, 3),
            CNBlockConfig(192, 384, 3),
            CNBlockConfig(384, 768, 9),
            CNBlockConfig(768, None, 3),
        ]
        super(ConvNext_MNIST, self).__init__(block_setting, stochastic_depth_prob=0.1, num_classes=10)


class ConvNext_CIFAR(ConvNeXt):
    # Eg. ConvNext_Small
    def __init__(self):
        block_setting = [
            CNBlockConfig(96, 192, 3),
            CNBlockConfig(192, 384, 3),
            CNBlockConfig(384, 768, 27),
            CNBlockConfig(768, None, 3),
        ]
        super(ConvNext_CIFAR, self).__init__(block_setting, stochastic_depth_prob=0.4, num_classes=100)


class ConvNext_Flowers(ConvNeXt):
    # Eg. ConvNext_Small
    def __init__(self):
        block_setting = [
            CNBlockConfig(96, 192, 3),
            CNBlockConfig(192, 384, 3),
            CNBlockConfig(384, 768, 27),
            CNBlockConfig(768, None, 3),
        ]
        super(ConvNext_Flowers, self).__init__(block_setting, stochastic_depth_prob=0.4, num_classes=102)