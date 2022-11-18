#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/18 17:34
# @Author : LvYan
# @Local : EscopeTech
import math
import os
import torch
from monai.data import load_decathlon_datalist, CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from monai.transforms import AsDiscrete
# from monai.visualize import plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter

from utils.utils import print_metrics
from nets.net import get_nets
from utils.loss import get_loss
from utils.optimizers import get_optimizer
from utils.transforms import train_transforms, val_transforms
from utils.config import argument
from loguru import logger


def run_train(args):
    logger.add(f"logs/train_{args.model_path}.log", level="INFO")
    writer = SummaryWriter(os.path.join(f"runs/{args.model_path}"))
    split_json = f"dataset_{args.labels}_{args.fold}.json"
    datasets = os.path.join(args.data_path, split_json)
    train_files = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")

    train_ds = CacheDataset(data=train_files, transform=train_transforms(args),
                            cache_rate=args.cache, num_workers=args.num_workers)
    val_ds = CacheDataset(data=val_files, transform=val_transforms(args),
                          cache_rate=args.cache, num_workers=args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model = get_nets(args).to(device)
    loss_function = get_loss(args)
    optimizer = get_optimizer(args, model)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_step, t_total=args.max_epoch)
    best_metric = 0
    for epoch in range(args.max_epoch):
        model.train()
        epoch_loss = 0
        for train_step, train_data in enumerate(train_loader):
            inputs, labels = train_data["image"].to(device), train_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # plot_2d_or_3d_image(inputs, epoch + 1, writer, index=0, tag="inputs")
            # plot_2d_or_3d_image(outputs, epoch + 1, writer, index=0, tag="outputs")
            # plot_2d_or_3d_image(labels, epoch + 1, writer, index=0, tag="labels")

        epoch_loss /= (train_step + 1)
        scheduler.step()
        writer.add_scalar("train_loss", epoch_loss, epoch + 1)
        writer.add_scalar("learning_rate", scheduler.lr_lambda(epoch), epoch + 1)
        print(f"Train Epoch {epoch + 1} Average Loss: {epoch_loss:.4f}")
        logger.info(f"Train Epoch {epoch + 1} Average Loss: {epoch_loss:.4f}")

        post_label = AsDiscrete(to_onehot=True, n_classes=args.num_classes)
        post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.num_classes)
        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_step, val_data in enumerate(val_loader):
                    inputs, labels = val_data["image"].to(device), val_data["label"].to(device)
                    outputs = sliding_window_inference(inputs, args.spatial_size, sw_batch_size=4,
                                                       overlap=0.5, mode="gaussian", predictor=model)
                    val_outputs = [post_pred(i) for i in decollate_batch(outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                metric_batch = dice_metric_batch.aggregate()
                dice_metric.reset()
                dice_metric_batch.reset()

                print(f"Val Epoch: {epoch + 1} Mean Dice: {metric:.4f}")
                logger.info(f"Val Epoch: {epoch + 1} Mean Dice: {metric:.4f}")
                print_metrics(args, epoch, metric_batch, writer, logger)

                os.makedirs(os.path.join("./models", args.model_path), exist_ok=True)
                torch.save(model.state_dict(), os.path.join("./models", args.model_path, f"final_model.pth"))
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join("./models", args.model_path, f"best_model.pth"))

    print(f"Train Completed, Best metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    logger.info(f"Train Completed, Best metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == '__main__':
    args = argument()
    run_train(args)

