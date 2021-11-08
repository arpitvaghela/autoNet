from darts import search, utils
import torch

if __name__ == "__main__":
    search.search("dog-cats", "custom", 32, epochs=1)
    # input_size, input_channels, n_classes, train_data = utils.get_data(
    #     "custom", "./dataset", cutout_length=0, validation=False
    # )
    # print(train_data)
    # train_loader = torch.utils.data.DataLoader(
    #     train_data,
    #     batch_size=2,
    #     num_workers=8,
    #     pin_memory=True,
    # )
    # for x, y in train_loader:
    #     print(x, y)
