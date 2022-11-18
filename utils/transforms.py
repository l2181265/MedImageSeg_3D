#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/18 15:06
# @Author : LvYan
# @Local : EscopeTech
from monai.transforms import (
    Compose,
    EnsureTyped,
    Invertd,
    AsDiscreted,
    LoadImaged,
    AddChanneld,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    ToTensord,
    RandShiftIntensityd,
    RandRotate90d,
    RandFlipd,
    RandCropByPosNegLabeld,
    EnsureChannelFirstd
)


def train_transforms(args):
    if args.dim == 3:
        train_trans = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=args.pixdim,
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAI"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=args.spatial_size,
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dim == 2:
        train_trans = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0.,
                    a_max=255.,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=args.spatial_size,
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )
    else:
        raise ValueError(f"Dim must be 2 or 3.")
    return train_trans


def val_transforms(args):
    if args.dim == 3:
        val_trans = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=args.pixdim,
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAI"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dim == 2:
        val_trans = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0.,
                    a_max=255.,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )
    else:
        raise ValueError(f"Dim must be 2 or 3.")
    return val_trans


def test_transforms(args):
    if args.dim == 3:
        test_trans = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(
                    keys=["image"],
                    pixdim=args.pixdim,
                    mode="bilinear",
                ),
                Orientationd(keys=["image"], axcodes="RAI"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dim == 2:
        test_trans = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0.,
                    a_max=255.,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )
    else:
        raise ValueError(f"Dim must be 2 or 3.")
    return test_trans


def post_transforms(args):
    if args.dim == 3:
        post_trans = Compose([
            EnsureTyped(keys="pred"),
            Invertd(
                keys="pred",
                transform=test_transforms(args),
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True, to_onehot=True, n_classes=args.num_classes),
            AsDiscreted(keys="label", to_onehot=True, n_classes=args.num_classes),
        ])
    elif args.dim == 2:
        post_trans = Compose([
            EnsureTyped(keys="pred"),
            AsDiscreted(keys="pred", argmax=True, to_onehot=True, n_classes=args.num_classes),
            AsDiscreted(keys="label", to_onehot=True, n_classes=args.num_classes),
        ])
    else:
        raise ValueError(f"Dim must be 2 or 3.")
    return post_trans