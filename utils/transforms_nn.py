#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/18 16:19
# @Author : LvYan
# @Local : EscopeTech
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged, \
    CropForegroundd, RandScaleCropd, RandRotated, RandAdjustContrastd, RandCropByPosNegLabeld, ToTensord, EnsureTyped, \
    Invertd, AsDiscreted


def train_transforms(args):

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
            RandScaleCropd(
                keys=["image", "label"],
                roi_scale=0.7,
                max_roi_scale=1.4,

            ),
            RandRotated(
                keys=["image", "label"],
                range_x=(-0.5235987755982988, 0.5235987755982988),
                range_y=(-0.5235987755982988, 0.5235987755982988),
                range_z=(-0.5235987755982988, 0.5235987755982988),
                prob=0.2,
            ),
            RandAdjustContrastd(
                keys=["image", "label"],
                prob=0.3,
                gamma=(0.7, 1.5)
            ),
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
            ToTensord(keys=["image", "label"]),
        ]
    )

    return train_trans


def val_transforms(args):

    val_trans = Compose(
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
                clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )

    return val_trans


def test_transforms(args):

    test_trans = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
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

    return test_trans


def post_transforms(args):

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

    return post_trans