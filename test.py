#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/18 21:36
# @Author : LvYan
# @Local : EscopeTech
import os
import pandas as pd
import numpy as np
import torch
import nibabel as nib
from monai.data import load_decathlon_datalist, CacheDataset, decollate_batch
from monai.handlers import from_engine
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader
from utils.metrics import evaluate, coronary_connect_region
from nets.net import get_nets
from utils.config import argument
from utils.transforms import test_transforms, post_transforms


def run_test(args):
    split_json = f"dataset_{args.labels}_{args.fold}.json"
    datasets = os.path.join(args.data_path, split_json)
    test_files = load_decathlon_datalist(datasets, True, "validation")
    test_ds = CacheDataset(data=test_files, transform=test_transforms(args), cache_rate=args.cache, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model_dir = os.path.join("./models", args.model_path, f"best_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model = get_nets(args).to(device)
    model.load_state_dict(torch.load(model_dir))

    M = list()
    model.eval()
    with torch.no_grad():
        for step, test_data in enumerate(test_loader):
            test_inputs, test_labels = test_data["image"].to(device), test_data["label"].to(device)
            test_data["pred"] = sliding_window_inference(test_inputs, args.spatial_size, 4, model)
            test_data = [post_transforms(args)(i) for i in decollate_batch(test_data)]
            test_outputs, test_labels = from_engine(["pred", "label"])(test_data)
            metrics = evaluate(test_outputs, test_labels)
            M.append(metrics)



if __name__ == '__main__':
    args = argument()
    run_test(args)