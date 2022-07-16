#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import argparse
import os
import random
import warnings
from loguru import logger

from PIL import Image
from airdet.dataset.transforms import build_transforms
import numpy as np
from airdet.utils import get_model_info, postprocess, vis
import cv2
from airdet.structures.image_list import to_image_list
from airdet.base_models.core.base_ops import SiLU
from airdet.utils.model_utils import replace_module

import torch
import torch.nn as nn

from airdet.config.base import parse_config
from airdet.detectors.detector_base import build_local_model, build_ddp_model
from airdet.base_models.heads import End2End
from io import BytesIO
import onnx

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


COCO_CLASSES = []
for i in range(80):
  COCO_CLASSES.append(str(i))
COCO_CLASSES = tuple(COCO_CLASSES)


def make_parser():
    parser = argparse.ArgumentParser("airdet eval")

    parser.add_argument(
        "-f",
        "--config_file",
        default='configs/airdet_s.py',
        type=str,
        help="pls input your config file",
    )
    parser.add_argument("-p", "--path", default='./assets/dog.jpg', type=str, help="path to image")
    parser.add_argument("-c", "--ckpt", default='./weights/airdet_s_ckpt.pth', type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="demo conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--end2end",
        default=False,
        action="store_true",
        help="End2End airdet model",
    )
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()

    origin_img = np.asarray(Image.open(args.path).convert("RGB"))
    device = "cuda"

    config = parse_config(args.config_file)
    config.merge(args.opts)

    transforms = build_transforms(config, is_train=False)

    img, _ = transforms(origin_img)
    img = to_image_list(img, config.dataset.size_divisibility).to(device)

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        config.testing.conf_threshold = args.conf
    if args.nms is not None:
        config.testing.nms_iou_threshold = args.nms

    model = build_local_model(config, device)
    logger.info("Model Summary: {}".format(get_model_info(model, (640, 640))))

    model.cuda(0)
    model.eval()

    ckpt_file = args.ckpt
    logger.info("loading checkpoint from {}".format(ckpt_file))
    loc = "cuda:{}".format(0)
    ckpt = torch.load(ckpt_file, map_location=loc)
    new_state_dict = {}
    for k, v in ckpt['model'].items():
        # if args.distributed:
        #     k = "module." + k
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model = replace_module(model, nn.SiLU, SiLU)
    logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    output_folder = os.path.join(config.miscs.output_dir, "demo")
    mkdir(output_folder)
    if args.end2end:
        model = End2End(model, max_obj=100, iou_thres=0.45, score_thres=0.25, device=device)
    output = model(img)
    nums, boxes, scores, classes = output

    zeros = torch.zeros((1,3,640,640),dtype=torch.float32).to(device)

    with BytesIO() as f:
        torch.onnx.export(
            model,zeros,f,
            input_names=['images'],
            output_names=['num_dets', 'det_boxes', 'det_scores', 'det_classes'],
        )
        f.seek(0)
        onnx_model = onnx.load(f)
    try:
        import onnxsim
        onnx_model, check = onnxsim.simplify(onnx_model,)
        assert check, 'assert check failed'
    except Exception as e:
        print(f"Error! Message is {e}")
    onnx.save(onnx_model,'end2end_airdet_s.onnx')


if __name__ == "__main__":
    main()
