from darts import search, utils
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms

if __name__ == "__main__":
    search.search("catdogs", "custom", 2, epochs=1)
    # search.search("mnist", "cifar10", 2, epochs=1)
    # input_size, input_channels, n_classes, train_data = utils.get_data(
    # "custom", "./dataset", cutout_length=0, validation=False
    # )
    # # ds = CIFAR10("./data/", train=True, download=True, transform=transforms.ToTensor())
    # train_loader = torch.utils.data.DataLoader(
    # train_data,
    # batch_size=2,
    # num_workers=8,
    # pin_memory=True,
    # )
    # for x, y in train_loader:
    # print(x.shape, y.shape)
    # break
