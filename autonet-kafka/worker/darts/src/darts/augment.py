""" Training augmented model """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from darts.config import AugmentConfig
import darts.utils as utils
from darts.models.augment_cnn import AugmentCNN


device = torch.device("cuda")


def augment_train(
    name: str,
    dataset: str,
    genotype: str,
    batch_size: int = 64,
    lr: float = 0.025,
    momentum: float = 0.9,
    weight_decay: float = 3e-4,
    grad_clip: float = 5.0,
    print_freq: int = 50,
    gpus: str = "0",
    epochs: int = 50,
    init_channels: int = 16,
    layers: int = 8,
    seed: int = 69,
    workers: int = 4,
    aux_weight: float = 0.4,
    cutout_length: int = 16,
    drop_path_prob: float = 0.2,
    projectid: str = "",
    dataid: str = "",
):
    if dataset == "custom" and (projectid == "" and dataid == ""):
        raise ValueError

    device = torch.device("cuda")
    data_path = "./data/"
    print(projectid)
    path = os.path.join("searchs", name)
    plot_path = os.path.join(path, "plots")
    # tensorboard
    writer = SummaryWriter(log_dir=os.path.join(path, "tb"))
    # writer.add_text('config', as_markdown(), 0)

    logger = utils.get_logger(os.path.join(path, "{}.log".format(name)))
    # print_params(logger.info)

    logger.info("Logger is set - training start")

    # set default gpu device id
    # torch.cuda.set_device(gpus[0])

    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data = utils.get_data(
        dataset,
        data_path,
        cutout_length,
        validation=True,
        projectid=projectid,
        dataid=dataid,
    )
    valid_data = train_data
    criterion = nn.CrossEntropyLoss().to(device)
    use_aux = aux_weight > 0.0
    model = AugmentCNN(
        input_size, input_channels, init_channels, n_classes, layers, use_aux, genotype
    )
    model = nn.DataParallel(model).to(device)

    # model size
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))

    # weights optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr, momentum=momentum, weight_decay=weight_decay
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    def train(train_loader, model, optimizer, criterion, epoch):
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        losses = utils.AverageMeter()

        cur_step = epoch * len(train_loader)
        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch {} LR {}".format(epoch, cur_lr))
        writer.add_scalar("train/lr", cur_lr, cur_step)

        model.train()

        for step, (X, y) in enumerate(train_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            optimizer.zero_grad()
            logits, aux_logits = model(X)
            loss = criterion(logits, y)
            if aux_weight > 0.0:
                loss += aux_weight * criterion(aux_logits, y)
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            maxk = min(5, n_classes - 1)
            prec1, prec5 = utils.accuracy(logits, y, topk=(1, maxk))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % print_freq == 0 or step == len(train_loader) - 1:
                logger.info(
                    "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
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
            "Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, epochs, top1.avg)
        )

    def validate(valid_loader, model, criterion, epoch, cur_step):
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        losses = utils.AverageMeter()

        model.eval()

        with torch.no_grad():
            for step, (X, y) in enumerate(valid_loader):
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                N = X.size(0)

                logits, _ = model(X)
                loss = criterion(logits, y)
                maxk = min(5, n_classes - 1)
                prec1, prec5 = utils.accuracy(logits, y, topk=(1, maxk))
                losses.update(loss.item(), N)
                top1.update(prec1.item(), N)
                top5.update(prec5.item(), N)

                if step % print_freq == 0 or step == len(valid_loader) - 1:
                    logger.info(
                        "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
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
            "Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, epochs, top1.avg)
        )

        return top1.avg

    best_top1 = 0.0
    # training loop
    for epoch in range(epochs):
        lr_scheduler.step()
        drop_prob = drop_path_prob * epoch / epochs
        model.module.drop_path_prob(drop_prob)

        # training
        train(train_loader, model, optimizer, criterion, epoch)

        # validation
        cur_step = (epoch + 1) * len(train_loader)
        top1 = validate(valid_loader, model, criterion, epoch, cur_step)

        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, path, is_best)

        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))


if __name__ == "__main__":
    config = AugmentConfig()
    augment_train(
        name=config.name,
        dataset=config.dataset,
        genotype=config.genotype,
        batch_size=config.batch_size,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
        print_freq=config.print_freq,
        gpus=config.gpus,
        epochs=config.epochs,
        init_channels=config.init_channels,
        layers=config.layers,
        seed=config.seed,
        workers=config.workers,
        aux_weight=config.aux_weight,
        cutout_length=config.cutout_length,
        drop_path_prob=config.drop_path_prob,
    )
