""" Utilities """
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np
import darts.preproc as preproc
from kafka_logger.handlers import KafkaLoggingHandler
from typing import Optional, Callable, Any, Tuple

# TODO: use env
bstrap_server = (
    dict(os.environ)["KAFKA_HOST"] if "KAFKA_HOST" in dict(os.environ) else ""
)
KAFKA_BOOTSTRAP_SERVER = bstrap_server
TOPIC = "log"


def custom_loader(path):
    img = dset.folder.default_loader(path)
    return img.resize((32, 32))


class CustomDataset(dset.ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = custom_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )
        data = []
        for path, _ in self:
            x = loader(path)
            x = x.resize((32, 32))
            data.append(np.asarray(x))
        self.data = np.stack(data)


def get_data(dataset, data_path, cutout_length, validation):
    """Get torchvision dataset"""
    dataset = dataset.lower()

    if dataset == "cifar10":
        dset_cls = dset.CIFAR10
        n_classes = 10
    elif dataset == "mnist":
        dset_cls = dset.MNIST
        n_classes = 10
    elif dataset == "fashionmnist":
        dset_cls = dset.FashionMNIST
        n_classes = 10
    elif dataset == "custom":
        dset_cls = CustomDataset
        n_classes = 2

    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
    if dataset == "custom":
        trn_data = dset_cls("./dataset", transform=trn_transform)
    else:
        trn_data = dset_cls(
            root=data_path, train=True, download=True, transform=trn_transform
        )

    # assuming shape is NHW or NHWC
    shape = trn_data.data.shape

    input_channels = 3 if len(shape) == 4 else 1
    assert shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]

    ret = [input_size, input_channels, n_classes, trn_data]
    if validation:  # append validation data
        ret.append(
            dset_cls(
                root=data_path, train=False, download=True, transform=val_transform
            )
        )

    return ret


def get_logger(file_path):
    """Make python logger"""
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger("darts")

    # TODO: move this part of code to worker
    kafak_handler_obj = KafkaLoggingHandler(
        KAFKA_BOOTSTRAP_SERVER, TOPIC, security_protocol="PLAINTEXT"
    )
    logger.addHandler(kafak_handler_obj)
    ##

    log_format = "%(asctime)s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%m/%d %I:%M:%S %p")
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """Compute parameter size in MB"""
    n_params = sum(
        np.prod(v.size())
        for k, v in model.named_parameters()
        if not k.startswith("aux_head")
    )
    return n_params / 1024.0 / 1024.0


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update statistics"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, "best.pth.tar")
        shutil.copyfile(filename, best_filename)
