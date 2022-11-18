#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/18 14:56
# @Author : LvYan
# @Local : EscopeTech
import torch
from monai.metrics import DiceMetric, ConfusionMatrixMetric, ROCAUCMetric
from sklearn.metrics import roc_auc_score
from config import argument

args = argument()
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
cfm_metric = ConfusionMatrixMetric(include_background=False,
                                   metric_name=("precision", "recall", "accuracy"),
                                   compute_sample=True)
auc_metric = ROCAUCMetric()


def evaluate(y_pred, y_true):
    dice_metric(y_pred=y_pred, y=y_true)
    dice_result = dice_metric.aggregate().item()
    dice_metric.reset()

    '''["sensitivity-TPR", "specificity-TNR", "precision-P", "negative predictive value", "miss rate-FNR", "fall out",
         "false discovery rate", "false omission rate", "prevalence threshold", "threat score", "accuracy",
         "balanced accuracy", "f1 score", "matthews correlation coefficient", "fowlkes mallows index", "informedness",
         "markedness"]'''

    cfm_metric(y_pred=y_pred, y=y_true)
    cfm_result = cfm_metric.aggregate()
    cfm_result = torch.stack(cfm_result).cpu().numpy()[:, 0]
    cfm_metric.reset()

    auc_metric(y_pred=y_pred, y=y_true)
    auc_result = auc_metric.aggregate().item()
    auc_result.reset()
    # auc_result = roc_auc_score(y_true=y_true[0][1].view(-1).cpu().numpy(), y_score=y_pred[0][1].view(-1).cpu().numpy())

    return dice_result, cfm_result, auc_result
