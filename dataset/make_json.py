#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/17 14:58
# @Author : LvYan
# @Local : EscopeTech
import json
import os
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# python make_json.py --path G:\code\dataset\3D_CTA144 --labels labels-2

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
arg = parser.add_argument
arg("--path", type=str, default="/dataset/3D_CTA167", help="")
arg("--labels", type=str, default="labels-10", help="")
arg("--folds", type=int, default=5, help="")
arg("--shuffle", type=bool, default=False, help="")
args = parser.parse_args()

image_path = f"{args.path}/images"
label_path = f"{args.path}/{args.labels}"
image_files = sorted([file for file in os.listdir(image_path)])
if args.shuffle:
    random.shuffle(image_files)

list_data = []
lengths = len(image_files)
print(lengths)
for i in range(lengths):
    image = image_files[i]
    label = image_files[i]

    dic = {
        "image": "images/" + image,
        "label": f"{args.labels}/" + label
    }
    list_data.append(dic)

folds = args.folds
for i in range(folds):
    data = {
        "name": "EscopeCT_Seg",
        "modality": {
            "0": "CTA"
        },
        # "labels": {
        #     "0": "background",
        #     "1": "foreground",
        # },
        "numTraining": int((folds-1) * lengths / folds),
        "numVal": int(lengths / folds),
        "training": list_data[:i * int(lengths / folds)] + list_data[(i+1) * int(lengths / folds):],
        "validation": list_data[i * int(lengths / folds):(i+1) * int(lengths / folds)]
    }

    file_name = os.path.join(args.path, f'dataset_{args.labels}_{i}.json')  # 通过扩展名指定文件存储的数据为json格式
    with open(file_name, 'w') as file_object:
        json.dump(data, file_object)