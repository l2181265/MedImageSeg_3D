#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/18 14:48
# @Author : LvYan
# @Local : EscopeTech
from monai.networks.layers import Act, Norm
from monai.networks.nets import UNet, VNet, BasicUNet, DynUNet, UNETR, SwinUNETR


def get_param(args):
    spatial_size = args.spatial_size
    spacing = args.pixdim
    kernels = []
    strides = []
    sizes, spacings = spatial_size, spacing
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return strides, kernels


def get_nets(args):
    global model
    if args.mode == "UNet":
        model = UNet(
            dimensions=args.dim,
            in_channels=args.in_channels,
            out_channels=args.num_classes,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            act=Act.PRELU,
            norm=Norm.INSTANCE,
            dropout=0.0,
        )
    elif args.mode == "VNet":
        model = VNet(
            spatial_dims=args.dim,
            in_channels=args.in_channels,
            out_channels=args.num_classes,
            act=("elu", {"inplace": True}),
            dropout_prob=0.5,
            dropout_dim=args.dim
        )
    elif args.mode == "BasicUNet":
        model = BasicUNet(
            dimensions=args.dim,
            in_channels=args.in_channels,
            out_channels=args.num_classes,
            features=(32, 64, 128, 256, 320),
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm=("instance", {"affine": True}),
            dropout=0.0,
            upsample="deconv"
        )
    elif args.mode == "DynUNet":
        strides, kernels = get_param(args)
        model = DynUNet(
            spatial_dims=args.dim,
            in_channels=args.in_channels,
            out_channels=args.num_classes,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name=("INSTANCE", {"affine": True}),
            deep_supervision=False,
            deep_supr_num=1,
            res_block=True)
    elif args.mode == "UNETR":
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.num_classes,
            img_size=args.spatial_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed='conv',
            norm_name='instance',
            conv_block=True,
            res_block=True,
            dropout_rate=0.0
        )
    elif args.mode == "SwinUNETR":
        model = SwinUNETR(
            in_channels=args.in_channels,
            out_channels=args.num_classes,
            img_size=args.spatial_size,
        )
    return model