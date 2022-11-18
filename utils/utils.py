#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/18 17:33
# @Author : LvYan
# @Local : EscopeTech


def print_metrics(args, epoch, metric_batch, writer, logger):
    metric_values_class = {}
    for m1 in range(1, args.num_classes):
        metric_values_class[f"{m1}"] = []

    for m2 in range(args.num_classes - 1):
        metric_num = metric_batch[m2].item()
        metric_values_class[f"{m2 + 1}"].append(metric_num)
        writer.add_scalar(f"dice_{m2 + 1}", metric_num, epoch + 1)
        print(f"Class {m2 + 1} Dice: {metric_num:.4f}")
        logger.info(f"Class {m2 + 1} Dice: {metric_num:.4f}")

