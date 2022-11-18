#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/18 14:50
# @Author : LvYan
# @Local : EscopeTech
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Sequence


def argument():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg("--data_path", type=str, default="./dataset", help="")
    arg("--labels", type=str, default="labels", help="")
    arg("--fold", type=int, default=0, help="")
    arg("--pixdim", type=Sequence[float], default=(0.35, 0.35, 0.5), help="")
    arg("--a_min", type=float, default=-200, help="")
    arg("--a_max", type=float, default=1000, help="")
    arg("--b_min", type=float, default=0.0, help="")
    arg("--b_max", type=float, default=1.0, help="")
    arg("--spatial_size", type=Sequence[int], default=(128, 128, 96), help="")
    arg("--cache", type=float, default=1.0, help="")
    arg("--num_workers", type=int, default=4, help="")
    arg("--batch_size", type=int, default=2, help="")
    arg("--val_batch_size", type=int, default=1, help="")

    arg("--dim", type=int, default=3, help="")
    arg("--num_classes", type=int, default=2, help="")
    arg("--learning_rate", type=float, default=1e-4, help="")
    arg("--weight_decay", type=float, default=1e-5, help="")
    arg("--momentum", type=float, default=0.99, help="")
    arg("--max_epoch", type=int, default=600, help="")
    arg("--model_path", type=str, default="./model", help="")
    arg("--pred_path", type=str, default="./preds", help="")
    arg("--mode", type=str, default="UNet", choices=["UNet", "VNet", "BasicUNet", "DynUNet", "UNETR", "SwinUNETR"])
    arg("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    arg("--loss_function", type=str, default="dicece",
        choices=["dice", "dicece", "dicefocal", "generalized", "GWD"])
    arg("--warmup_step", type=int, default=10, help="")
    arg("--val_interval", type=int, default=5, help="")
    arg("--scheduler", action="store_true", help="")
    arg("--in_channels", type=int, default=1, help="")
    arg("--continue_train", action="store_true", help="")
    args = parser.parse_args()
    return args