""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import darts.config as cf
from darts.config import SearchConfig
import darts.utils as utils
from darts.models.search_cnn import SearchCNNController
from darts.architect import Architect
from darts.visualize import plot
import darts.genotypes as gt
from darts.models import ops
from typing import List


def set_primitives(primitives: List[str]):
    gt.PRIMITIVES = primitives


def set_default_primitives():
    gt.PRIMITIVES = [
        "max_pool_3x3",
        "avg_pool_3x3",
        "skip_connect",  # identity
        "sep_conv_3x3",
        "sep_conv_5x5",
        "sep_conv_7x7",
        "dil_conv_3x3",
        "dil_conv_5x5",
        "none",
    ]


def search(
    name: str,
    dataset: str,
    batch_size: int = 64,
    w_lr: float = 0.025,
    w_lr_min: float = 0.001,
    w_momentum: float = 0.9,
    w_weight_decay: float = 3e-4,
    w_grad_clip: float = 5.0,
    print_freq: int = 50,
    gpus: str = "0",
    epochs: int = 50,
    init_channels: int = 16,
    layers: int = 8,
    seed: int = 69,
    workers: int = 4,
    alpha_lr: float = 3e-4,
    alpha_weight_decay: float = 1e-3,
    projectid: str = "",
    dataid: str = "",
):
    """TODO: Add docstring"""
    if dataset == "custom" and (projectid == "" and dataid == ""):
        raise ValueError

    # set primitive
    if not hasattr(gt, "PRIMITIVES"):
        set_default_primitives()

    # tensorboard
    if isinstance(gpus, str):
        gpus = cf.parse_gpus(gpus)

    device = torch.device("cuda")
    data_path = "./data/"
    print(projectid)
    path = os.path.join("searchs", name)
    plot_path = os.path.join(path, "plots")

    writer = SummaryWriter(log_dir=os.path.join(path, "tb"))
    # writer.add_text("config", as_markdown(), 0)
    logger = utils.get_logger(os.path.join(path, "{}.log".format(name)))

    def train(
        train_loader,
        valid_loader,
        model,
        architect,
        w_optim,
        alpha_optim,
        lr,
        epoch,
        print_freq,
        logger,
        epochs,
    ):
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        losses = utils.AverageMeter()

        cur_step = epoch * len(train_loader)
        writer.add_scalar("train/lr", lr, cur_step)

        model.train()

        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(
            zip(train_loader, valid_loader)
        ):
            trn_X, trn_y = (
                trn_X.to(device, non_blocking=True),
                trn_y.to(device, non_blocking=True),
            )
            val_X, val_y = (
                val_X.to(device, non_blocking=True),
                val_y.to(device, non_blocking=True),
            )
            N = trn_X.size(0)

            # phase 2. architect step (alpha)
            alpha_optim.zero_grad()
            architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
            alpha_optim.step()

            # phase 1. child network step (w)
            w_optim.zero_grad()
            logits = model(trn_X)
            loss = model.criterion(logits, trn_y)
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(model.weights(), w_grad_clip)
            w_optim.step()
            maxk = min(5, n_classes - 1)
            prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, maxk))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % print_freq == 0 or step == len(train_loader) - 1:
                logger.info(
                    "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1,
                        epochs,
                        step,
                        len(train_loader) - 1,
                        losses=losses,
                        top1=top1,
                        top5=top5,
                    )
                )

            writer.add_scalar("train/loss", loss.item(), cur_step)
            writer.add_scalar("train/top1", prec1.item(), cur_step)
            writer.add_scalar("train/top5", prec5.item(), cur_step)
            cur_step += 1

        logger.info(
            "Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, epochs, top1.avg)
        )

    def validate(valid_loader, model, epoch, cur_step):
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        losses = utils.AverageMeter()

        model.eval()

        with torch.no_grad():
            for step, (X, y) in enumerate(valid_loader):
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                N = X.size(0)

                logits = model(X)
                loss = model.criterion(logits, y)
                maxk = min(5, n_classes - 1)
                prec1, prec5 = utils.accuracy(logits, y, topk=(1, maxk))
                losses.update(loss.item(), N)
                top1.update(prec1.item(), N)
                top5.update(prec5.item(), N)

                if step % print_freq == 0 or step == len(valid_loader) - 1:
                    logger.info(
                        "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                        "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                            epoch + 1,
                            epochs,
                            step,
                            len(valid_loader) - 1,
                            losses=losses,
                            top1=top1,
                            top5=top5,
                        )
                    )

        writer.add_scalar("val/loss", losses.avg, cur_step)
        writer.add_scalar("val/top1", top1.avg, cur_step)
        writer.add_scalar("val/top5", top5.avg, cur_step)

        logger.info(
            "Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, epochs, top1.avg)
        )

        return top1.avg

    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(gpus[0])

    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data = utils.get_data(
        dataset,
        data_path,
        cutout_length=0,
        validation=False,
        projectid=projectid,
        dataid=dataid,
    )

    net_crit = nn.CrossEntropyLoss().to(device)
    model = SearchCNNController(
        input_channels,
        init_channels,
        n_classes,
        layers,
        net_crit,
        device_ids=gpus,
    )
    model = model.to(device)

    # weights optimizer
    w_optim = torch.optim.SGD(
        model.weights(),
        w_lr,
        momentum=w_momentum,
        weight_decay=w_weight_decay,
    )
    # alphas optimizer
    alpha_optim = torch.optim.Adam(
        model.alphas(),
        alpha_lr,
        betas=(0.5, 0.999),
        weight_decay=alpha_weight_decay,
    )

    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=workers,
        pin_memory=True,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, epochs, eta_min=w_lr_min
    )
    architect = Architect(model, w_momentum, w_weight_decay)

    # training loop
    best_top1 = 0.0
    for epoch in range(epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)

        # training
        train(
            train_loader,
            valid_loader,
            model,
            architect,
            w_optim,
            alpha_optim,
            lr,
            epoch,
            print_freq,
            logger,
            epochs,
        )

        # validation
        cur_step = (epoch + 1) * len(train_loader)
        top1 = validate(valid_loader, model, epoch, cur_step)

        # log
        # genotype
        genotype = model.genotype()
        best_genotype = genotype
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        plot_path = os.path.join(plot_path, "EP{:02d}".format(epoch + 1))
        caption = "Epoch {}".format(epoch + 1)
        plot(genotype.normal, plot_path + "-normal", caption)
        plot(genotype.reduce, plot_path + "-reduce", caption)

        # save
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, path, is_best)
        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))
    return best_genotype


if __name__ == "__main__":
    config = SearchConfig()

    # config.print_params(logger.info)
    search(
        name=config.name,
        dataset=config.dataset,
        batch_size=config.batch_size,
        w_lr=config.w_lr,
        w_lr_min=config.w_lr_min,
        w_momentum=config.w_momentum,
        w_weight_decay=config.w_weight_decay,
        w_grad_clip=config.w_grad_clip,
        print_freq=config.print_freq,
        gpus=config.gpus,
        epochs=config.epochs,
        init_channels=config.init_channels,
        layers=config.layers,
        seed=config.seed,
        workers=config.workers,
        alpha_lr=config.alpha_lr,
        alpha_weight_decay=config.alpha_weight_decay,
    )
