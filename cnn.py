import torch

from dataset_bci_iv_2a.dataset import BciIvDatasetFactory
from torch.utils.data import DataLoader


if __name__ == "__main__":
    trainset, testset = BciIvDatasetFactory.create(1, 20, 10)
    batch_size: int = 4

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    trainiter = iter(trainloader)

    samples, labels = next(trainiter)

    print(samples)
    print(labels)
