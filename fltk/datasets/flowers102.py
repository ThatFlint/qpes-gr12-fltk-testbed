from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets
from torchvision import transforms

from fltk.datasets.dataset import Dataset


class Flowers102Dataset(Dataset):
    """
    Flowers102 Dataset implementation for Distributed learning experiments.
    """

    DEFAULT_TRANSFORM = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=64, pad_if_needed=True),
        transforms.Resize(size=28),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # RGB mean & std estied on ImageNet
                             [0.229, 0.224, 0.225])
    ])

    def __init__(self, config, learning_param, rank: int = 0, world_size: int = None):
        super(Flowers102Dataset, self).__init__(config, learning_param, rank, world_size)

    def load_train_dataset(self, rank: int = 0, world_size: int = None):
        train_dataset = datasets.Flowers102(root=self.config.get_data_path(), split='train',download=True,transform=self.DEFAULT_TRANSFORM )
        sampler = DistributedSampler(train_dataset, rank=rank,
                                     num_replicas=self.world_size) if self.world_size else None
        train_loader = DataLoader(train_dataset, batch_size=self.learning_params.batch_size, sampler=sampler,
                                  shuffle=(sampler is None))

        return train_loader

    def load_test_dataset(self):

        test_dataset = datasets.Flowers102(root=self.config.get_data_path(), split='test',download=True,transform=self.DEFAULT_TRANSFORM )
        sampler = DistributedSampler(test_dataset, rank=self.rank,
                                     num_replicas=self.world_size) if self.world_size else None
        test_loader = DataLoader(test_dataset, batch_size=self.learning_params.test_batch_size, sampler=sampler)
        return test_loader
