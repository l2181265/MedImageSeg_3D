#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/18 15:04
# @Author : LvYan
# @Local : EscopeTech
import torch


def get_optimizer(args, model):
    global optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
                                    momentum=args.momentum)
    return optimizer