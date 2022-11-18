#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/18 14:53
# @Author : LvYan
# @Local : EscopeTech
import numpy as np
import torch
from monai.losses import (
    DiceLoss,
    DiceFocalLoss,
    GeneralizedDiceLoss,
    GeneralizedWassersteinDiceLoss,
    DiceCELoss
)
from monai.networks import one_hot


def get_loss(args):
    loss_dice = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
    loss_dicece = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    loss_dicefocal = DiceFocalLoss(include_background=False, to_onehot_y=True, softmax=True)
    loss_diceG = GeneralizedDiceLoss(include_background=False, to_onehot_y=True, softmax=True)
    loss_diceGWD = GeneralizedDiceLoss(include_background=False, to_onehot_y=True, softmax=True)

    loss_function = [loss_dice, loss_dicece, loss_dicefocal, loss_diceG, loss_diceGWD]

    if args.loss_function == 'dice':
        loss_function = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
    elif args.loss_function == 'dicece':
        loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    elif args.loss_function == 'dicefocal':
        loss_function = DiceFocalLoss(include_background=False, to_onehot_y=True, softmax=True)
    elif args.loss_function == 'generalized':
        loss_function = GeneralizedDiceLoss(include_background=False, to_onehot_y=True, softmax=True)
    elif args.loss_function == 'GWD':
        m_tree = np.array([[0.0, 1.0, 1.0, 1.0],
                           [1.0, 0.0, 0.4, 0.6],
                           [1.0, 0.4, 0.0, 0.8],
                           [1.0, 0.6, 0.8, 0.0]])
        loss_function = GeneralizedWassersteinDiceLoss(dist_matrix=m_tree, weighting_mode='default', reduction='mean')
    return loss_function


def calculate_loss(loss_function, y_pred, y_true):
    if isinstance(loss_function, list):
        n_pred_ch = y_pred.shape[1]
        y_pred = torch.softmax(y_pred, 1)
        y_true = one_hot(y_true, num_classes=n_pred_ch)
        loss1 = loss_function[0](y_pred[:, :2, ...], y_true[:, :2, ...])
        loss2 = loss_function[2](y_pred[:, 0:3:2, ...], y_true[:, 0:3:2, ...])
        loss3 = loss_function[1](y_pred[:, 0:4:3, ...], y_true[:, 0:4:3, ...])
        loss = [loss1, loss2, loss3]
        return loss
    else:
        loss = loss_function(y_pred, y_true)
        return loss

