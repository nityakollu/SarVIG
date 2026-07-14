# 2022.06.17-Changed for training ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# !/usr/bin/env python
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import warnings

warnings.filterwarnings('ignore')
import argparse
import time
import yaml
import os
import logging

from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import resolve_data_config, Mixup, FastCollateMixup, AugMixDataset  # , create_loader
from torchvision.datasets import ImageFolder as Dataset
from timm.models import create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

from data.myloader import create_loader
#import pyramid_vig not using this model
import vig
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

from data.patch_dataset import PatchDataset
#expected structure for patch datatsets:
#dataset_eval/train/sar/images, datatset/val/sar/images, dataset_eval/train_masks/ice or ocean or no_data/images,dataset_eval/val_masks/ice or ocean or no_data/images

from data.unlabeled_sar_dataset import UnlabeledSarDataset

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('data', metavar='DIR',
                    help='path to dataset_eval')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset_eval')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset_eval')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')
parser.add_argument('--grid-size', type=int, default=14, choices=[7, 14, 28, 56, 112],
                    help='Output grid size for ViG stem')
parser.add_argument('--classifier-mode', type=str, default='image', choices=['image', 'patch', 'self_supervised'],
                    help='Classification mode: image-level or patch-level or self supervised')
parser.add_argument('--in-chans', type=int, default=3, choices=[1, 3],
                    help='Number of input channels expected by the model')
parser.add_argument('--train-only', action='store_true', default=False,
                    help='Train only, skip validation during epochs.')
parser.add_argument('--skip-val', action='store_true', default=False,
                    help='Do not require validation dataset_eval.')
#freeze modes for finetuning
parser.add_argument('--freeze-mode',default='none',type=str,choices=['none', 'partial', 'backbone'],help='Freezing strategy: none, partial, or backbone')
parser.add_argument('--freeze-blocks',default=0,type=int,help='For partial freeze: number of early ViG blocks to freeze')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--repeated-aug', action='store_true')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)')
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
# for huawei cloud
parser.add_argument("--init_method", default='env://', type=str)
parser.add_argument("--train_url", type=str)
# newly added
parser.add_argument('--attn_ratio', type=float, default=1.,
                    help='attention ratio')
parser.add_argument("--pretrain_path", default=None, type=str)
parser.add_argument("--evaluate", action='store_true', default=False,
                    help='whether evaluate the model')

def prefix_dict(d, prefix):
    return {f'{prefix}{k}': v for k, v in d.items()}

def flatten_dict(d, parent_key=''):
    items = {}
    for k, v in d.items():
        new_key = f'{parent_key}_{k}' if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items

"""
Helper for freeze options. none trains everything, backbone freezes everything execpt classification head
partial freezes stem and first n blocks where n is specified by freeze_blocks 
"""
def freeze_mode(model, freeze_mode='none', freeze_blocks=0):
    freeze_mode = freeze_mode.lower()

    for param in model.parameters():
        param.requires_grad = True

    if freeze_mode == 'none':
        print("Freeze mode: none, training all parameters.")

    elif freeze_mode == 'backbone':
        print("Freeze mode: backbone, freezing stem, pos_embed, and backbone. Training prediction head only.")

        for name, param in model.named_parameters():
            param.requires_grad = False

        #patch classification prediction head
        for param in model.prediction.parameters():
            param.requires_grad = True

    elif freeze_mode == 'partial':
        print(f"Freeze mode: partial, freezing stem and first {freeze_blocks} backbone blocks.")

        #freeze the stem
        for param in model.stem.parameters():
            param.requires_grad = False

        #freeze pos embed, it's part of backbone
        model.pos_embed.requires_grad = False

        #freeze n blocks in backbone
        for i in range(min(freeze_blocks, model.n_blocks)):
            for param in model.backbone[i].parameters():
                param.requires_grad = False
            print(f"Frozen: backbone[{i}]")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params

    print(f"Trainable params: {trainable_params:,}")
    print(f"Frozen params: {frozen_params:,}")
    print(f"Total params: {total_params:,}")

    return model

"""
Helper to add experiment info and metrics to CSV for tracking
"""
def append_experiment_output_csv(csv_path, config_dict, metrics_dict, extra_dict=None):
    row = {}

    if extra_dict is not None:
        row.update(extra_dict)

    #prefix the config keys so they are separate from metrics
    flat_config = flatten_dict(config_dict)
    for k, v in flat_config.items():
        row[f'cfg_{k}'] = v

    for k, v in metrics_dict.items():
        row[k] = v

    csv_path = Path(csv_path)

    #if file does not exist yet, write it directly
    if not csv_path.exists():
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    # Read existing rows so we can expand columns if needed
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        old_rows = list(reader)
        old_fields = reader.fieldnames if reader.fieldnames is not None else []

    # union of old + new columns, preserving prior order
    new_fields = list(old_fields)
    for key in row.keys():
        if key not in new_fields:
            new_fields.append(key)

    old_rows.append(row)

    # rewrite full CSV with expanded columns if necessary
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        for r in old_rows:
            writer.writerow(r)

def save_patch_overlay_figure(
    input_tensor,
    pred_patch_map,
    save_path,
    ignore_index=-100,
    alpha=0.35,
):
    img = input_tensor.detach().cpu().permute(1, 2, 0).numpy()

    img = (img * 0.5) + 0.5
    img = np.clip(img, 0, 1)

    H, W, C = img.shape

    if hasattr(pred_patch_map, "detach"):
        pred = pred_patch_map.detach().cpu().numpy()
    else:
        pred = np.asarray(pred_patch_map)

    Gh, Gw = pred.shape

    if H % Gh != 0 or W % Gw != 0:
        raise ValueError(
            f"Image size {(H, W)} is not divisible by patch map size {(Gh, Gw)}"
        )

    patch_h = H // Gh
    patch_w = W // Gw
    overlay = np.zeros((H, W, 4), dtype=np.float32)

    for r in range(Gh):
        for c in range(Gw):
            label = pred[r, c]

            y1 = r * patch_h
            y2 = (r + 1) * patch_h
            x1 = c * patch_w
            x2 = (c + 1) * patch_w

            if label == 0:
                # ocean  is green
                overlay[y1:y2, x1:x2, :3] = [0.0, 1.0, 0.0]
                overlay[y1:y2, x1:x2, 3] = alpha
            elif label == 1:
                # ice  is blue
                overlay[y1:y2, x1:x2, :3] = [0.0, 0.0, 1.0]
                overlay[y1:y2, x1:x2, 3] = alpha
            elif label == 2:
                # boundary areas are yellow
                overlay[y1:y2, x1:x2, :3] = [1.0, 1.0, 0.0]
                overlay[y1:y2, x1:x2, 3] = alpha
            elif label == 3:
                # small ice is purple
                overlay[y1:y2, x1:x2, :3] = [0.5, 0.0, 0.5]
                overlay[y1:y2, x1:x2, 3] = alpha
            elif label == ignore_index:
                # no data red
                overlay[y1:y2, x1:x2, :3] = [1.0, 0.0, 0.0]
                overlay[y1:y2, x1:x2, 3] = alpha
            else:
                #unexpected things will be white
                overlay[y1:y2, x1:x2, :3] = [1.0, 1.0, 1.0]
                overlay[y1:y2, x1:x2, 3] = alpha
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(img)
    axes[1].imshow(overlay)
    axes[1].set_title("Predicted Patch Labels Overlay")
    axes[1].axis("off")

    for r in range(Gh + 1):
        y = r * patch_h
        axes[1].axhline(y, color="white", linewidth=0.2, alpha=0.5)
    for c in range(Gw + 1):
        x = c * patch_w
        axes[1].axvline(x, color="white", linewidth=0.2, alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_prediction_and_target_overlays(
        input_tensor,
        target_patch_map,
        pred_patch_map,
        save_path,
        ignore_index=-100,
        alpha=0.35,
):
    img = input_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5) + 0.5
    img = np.clip(img, 0, 1)

    if hasattr(target_patch_map, "detach"):
        target = target_patch_map.detach().cpu().numpy()
    else:
        target = np.asarray(target_patch_map)

    if hasattr(pred_patch_map, "detach"):
        pred = pred_patch_map.detach().cpu().numpy()
    else:
        pred = np.asarray(pred_patch_map)

    H, W, C = img.shape
    Gh, Gw = pred.shape
    patch_h = H // Gh
    patch_w = W // Gw

    def make_overlay(label_map):
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        for r in range(Gh):
            for c in range(Gw):
                label = label_map[r, c]
                y1 = r * patch_h
                y2 = (r + 1) * patch_h
                x1 = c * patch_w
                x2 = (c + 1) * patch_w

                if label == 0:
                    #ocean is green
                    overlay[y1:y2, x1:x2, :3] = [0.0, 1.0, 0.0]
                    overlay[y1:y2, x1:x2, 3] = alpha
                elif label == 1:
                    #ice is blue
                    overlay[y1:y2, x1:x2, :3] = [0.0, 0.0, 1.0]
                    overlay[y1:y2, x1:x2, 3] = alpha
                elif label == 2:
                    #boundary area yellow
                    overlay[y1:y2, x1:x2, :3] = [1.0, 1.0, 0.0]
                    overlay[y1:y2, x1:x2, 3] = alpha
                elif label == 3:
                    # small ice is purple
                    overlay[y1:y2, x1:x2, :3] = [0.5, 0.0, 0.5]
                    overlay[y1:y2, x1:x2, 3] = alpha
                elif label == ignore_index:
                    #no data red
                    overlay[y1:y2, x1:x2, :3] = [1.0, 0.0, 0.0]
                    overlay[y1:y2, x1:x2, 3] = alpha
                else:
                    #unexpected things white for debug
                    overlay[y1:y2, x1:x2, :3] = [1.0, 1.0, 1.0]
                    overlay[y1:y2, x1:x2, 3] = alpha
        return overlay

    target_overlay = make_overlay(target)
    pred_overlay = make_overlay(pred)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(img)
    axes[1].imshow(target_overlay)
    axes[1].set_title("Actual Patch Labels Overlay")
    axes[1].axis("off")

    axes[2].imshow(img)
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Predicted Patch Labels Overlay")
    axes[2].axis("off")

    for ax in [axes[1], axes[2]]:
        for r in range(Gh + 1):
            y = r * patch_h
            ax.axhline(y, color="white", linewidth=0.2, alpha=0.5)
        for c in range(Gw + 1):
            x = c * patch_w
            ax.axvline(x, color="white", linewidth=0.2, alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _parse_args():
    cfg = {}
    config_path = ''
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        config_path = args_config.config
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text, cfg, config_path

def main():
    setup_default_logging()
    args, args_text, config_only, config_path = _parse_args()

    #use proper metric for each mode
    if args.classifier_mode == 'patch' and args.eval_metric == 'top1':
        args.eval_metric = 'patch_acc'
    elif args.classifier_mode == 'self_supervised' and args.eval_metric == 'top1':
        args.eval_metric = 'loss'

    args.prefetcher = not args.no_prefetcher
    if args.classifier_mode == 'self_supervised':
        args.prefetcher = False
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            _logger.warning(
                'Using more than one GPU per process in distributed mode is not allowed.Setting num_gpu to 1.')
            args.num_gpu = 1

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank,
                                             world_size=args.world_size)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    assert args.rank >= 0

    if args.distributed:
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on %d GPUs.' % args.num_gpu)

    torch.manual_seed(args.seed + args.rank)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        checkpoint_path=args.initial_checkpoint,
        img_size=args.img_size if args.img_size is not None else 224,
        grid_size=args.grid_size,
        in_chans=args.in_chans,
        classifier_mode=args.classifier_mode,
    )

    ################## pretrain ############
    if args.pretrain_path is not None:
        print('Loading:', args.pretrain_path)
        state_dict = torch.load(args.pretrain_path)

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        from vig import load_pretrained_pos_embed
        load_pretrained_pos_embed(model, state_dict)

        print('Pretrain weights loaded.')
    ############### Freeeeeze####################
    model = freeze_mode(model, freeze_mode=args.freeze_mode, freeze_blocks=args.freeze_blocks)
    ################### flops #################
    print(model)
    if hasattr(model, 'default_cfg'):
        default_cfg = model.default_cfg
        input_size = [1] + list(default_cfg['input_size'])
    else:
        input_size = [1, 3, 224, 224]
    input = torch.randn(input_size)  # .cuda()

    from torchprofile import profile_macs
    model.eval()
    macs = profile_macs(model, input)
    model.train()
    print('model flops:', macs, 'input_size:', input_size)
    ##########################################

    if args.local_rank == 0:
        _logger.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    use_amp = None
    if args.amp:
        # for backwards compat, `--amp` arg tries apex before native amp
        if has_apex:
            args.apex_amp = True
        elif has_native_amp:
            args.native_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    if args.num_gpu > 1:
        if use_amp == 'apex':
            _logger.warning(
                'Apex AMP does not work well with nn.DataParallel, disabling. Use DDP or Torch AMP.')
            use_amp = None
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        assert not args.channels_last, "Channels last not supported with DP, use DDP."
    else:
        model.cuda()
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)

    optimizer = create_optimizer(args, model)

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume)

    if args.distributed:
        if args.sync_bn:
            assert not args.split_bn
            try:
                if has_apex and use_amp != 'native':
                    # Apex SyncBN preferred unless native amp is activated
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if args.local_rank == 0:
                    _logger.info(
                        'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                        'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
            except Exception as e:
                _logger.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    dataset_train = None
    loader_train = None

    # Only require/load training data if we are actually training
    if not args.evaluate:
        train_dir = os.path.join(args.data, 'train')
        if not os.path.exists(train_dir):
            _logger.error('Training folder does not exist at: {}'.format(train_dir))
            exit(1)

        if args.classifier_mode == 'patch':
            dataset_train = PatchDataset(
                train_dir,
                grid_size=args.grid_size,
                img_size=args.img_size if args.img_size is not None else 224,
                in_chans=args.in_chans,
                num_classes=args.num_classes
            )
        elif args.classifier_mode == 'self_supervised':
            dataset_train = UnlabeledSarDataset(
                train_dir,
                img_size=args.img_size if args.img_size is not None else 224,
                in_chans=args.in_chans
            )
        else:
            dataset_train = Dataset(train_dir)

    collate_fn = None
    mixup_fn = None

    mixup_active = (
            args.classifier_mode != 'self_supervised'
            and (args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None)
    )

    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes
        )
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    if num_aug_splits > 1 and args.classifier_mode != 'self_supervised':
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    loader_eval = None
    validate_loss_fn = None

    if not args.evaluate:
        train_interpolation = args.train_interpolation
        if args.no_aug or not train_interpolation:
            train_interpolation = data_config['interpolation']

        loader_train = create_loader(
            dataset_train,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            no_aug=args.no_aug,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            re_split=args.resplit,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_splits=num_aug_splits,
            interpolation=train_interpolation,
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=args.pin_mem,
            use_multi_epochs_loader=args.use_multi_epochs_loader,
            repeated_aug=args.repeated_aug
        )

    need_validation = (not args.train_only) or args.evaluate

    if need_validation:
        eval_dir = os.path.join(args.data, 'val')
        if not os.path.isdir(eval_dir):
            eval_dir = os.path.join(args.data, 'validation')
            if not os.path.isdir(eval_dir):
                _logger.error('Validation folder does not exist at: {}'.format(eval_dir))
                exit(1)
        if args.classifier_mode == 'patch':
            dataset_eval = PatchDataset(
                eval_dir,
                grid_size=args.grid_size,
                img_size=args.img_size if args.img_size is not None else 224,
                in_chans=args.in_chans,
                num_classes=args.num_classes,
            )
        else:
            dataset_eval = Dataset(eval_dir)

        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size_multiplier * args.batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
        )

    if args.classifier_mode == 'patch':
        train_loss_fn = nn.CrossEntropyLoss(ignore_index=-100).cuda()
        validate_loss_fn = nn.CrossEntropyLoss(ignore_index=-100).cuda() if need_validation else None

    elif args.classifier_mode == 'self_supervised':
        train_loss_fn = nn.MSELoss().cuda()
        validate_loss_fn = nn.MSELoss().cuda() if need_validation else None

    else:
        if args.jsd:
            assert num_aug_splits > 1  # JSD only valid with aug splits set
            train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
        elif mixup_active:
            # smoothing is handled with mixup target transform
            train_loss_fn = SoftTargetCrossEntropy().cuda()
        elif args.smoothing:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
        else:
            train_loss_fn = nn.CrossEntropyLoss().cuda()

        validate_loss_fn = nn.CrossEntropyLoss().cuda() if need_validation else None

    output_dir = args.output if args.output else "./output"

    if args.evaluate:
        eval_metrics = validate(
            model, loader_eval, validate_loss_fn, args,
            amp_autocast=amp_autocast,
            output_dir=output_dir,
            epoch = "eval",
            vis_idx = [0, 2, 7, 12]
        )
        print(eval_metrics)

        if args.local_rank == 0:
            append_experiment_output_csv(
                'exp_output.csv',
                config_only,
                prefix_dict(eval_metrics, 'eval_'),
                extra_dict={
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'run_type': 'evaluate',
                    'classifier_mode': args.classifier_mode,
                    'config_file': config_path
                }
            )

        return

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = ''
    eval_metrics = None
    final_train_metrics = None
    final_eval_metrics = None
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model,
            str(data_config['input_size'][-1])
        ])
        output_dir = get_outdir(output_base, 'train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler,
                model_ema=model_ema, mixup_fn=mixup_fn
            )
            final_train_metrics = train_metrics

            if need_validation:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    if args.local_rank == 0:
                        _logger.info("Distributing BatchNorm running means and vars")
                    distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

                eval_metrics = validate(
                    model, loader_eval, validate_loss_fn, args,
                    amp_autocast=amp_autocast,
                    output_dir=output_dir,
                    epoch = epoch,
                    vis_idx = [0, 4]
                )
                if args.local_rank == 0:
                    print(eval_metrics)
                final_eval_metrics = eval_metrics

                if model_ema is not None and not args.model_ema_force_cpu:
                    if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                        distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                    ema_eval_metrics = validate(
                        model_ema.ema, loader_eval, validate_loss_fn, args,
                        amp_autocast=amp_autocast,
                        log_suffix=' (EMA)',
                        output_dir=output_dir
                    )
                    eval_metrics = ema_eval_metrics
                    final_eval_metrics = eval_metrics

                if lr_scheduler is not None:
                    lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

                update_summary(
                    epoch, train_metrics, eval_metrics,
                    os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None
                )

                if saver is not None:
                    save_metric = eval_metrics[eval_metric]
                    best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            else:
                if lr_scheduler is not None:
                    # step LR for next epoch
                    lr_scheduler.step(epoch + 1)

                update_summary(
                    epoch, train_metrics, OrderedDict(),
                    os.path.join(output_dir, 'summary.csv'),
                    write_header=(epoch == start_epoch)
                )
                final_eval_metrics = OrderedDict()

                if saver is not None:
                    # save proper checkpoint with eval metric
                    save_metric = train_metrics['loss']
                    best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if args.local_rank == 0:
        merged_metrics = {}

        if final_train_metrics is not None:
            merged_metrics.update(prefix_dict(final_train_metrics, 'train_'))

        if final_eval_metrics is not None:
            merged_metrics.update(prefix_dict(final_eval_metrics, 'eval_'))

        append_experiment_output_csv(
            'exp_output.csv',
            config_only,
            merged_metrics,
            extra_dict={
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'run_type': 'train',
                'classifier_mode': args.classifier_mode,
                'config_file': config_path
            }
        )
    if best_metric is not None:
        _logger.info('*** Best {0}: {1} (epoch {2})'.format(eval_metric, best_metric, best_epoch))


def train_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, batch in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        input, target = batch

        if not args.prefetcher:
            input = input.cuda()
            if target is not None:
                target = target.cuda()
            if mixup_fn is not None and target is not None:
                input, target = mixup_fn(input, target)

        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input)

            if batch_idx == 0 and epoch % 1 == 0:
                import torchvision.utils as vutils
                import os

                os.makedirs(os.path.join(output_dir, "recon"), exist_ok=True)

                vutils.save_image(input,
                                  os.path.join(output_dir, f"recon/input_epoch{epoch}.png"),
                                  normalize=True)

                vutils.save_image(output,
                                  os.path.join(output_dir, f"recon/recon_epoch{epoch}.png"),
                                  normalize=True)

            if args.classifier_mode == "self_supervised":
                loss = loss_fn(output, input)
            else:
                loss = loss_fn(output, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(), create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx if last_idx > 0 else 100.,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def get_metrics_from_confusion_matrix(cm):
    cm = cm.float()
    # rows = true class, columns = predicted class.
    num_classes = cm.shape[0]
    tp = torch.diag(cm)
    pred_per_class = cm.sum(dim=0)
    true_per_class = cm.sum(dim=1)
    prec = torch.where(pred_per_class > 0, tp / pred_per_class, torch.zeros_like(tp)) #precision is the number of true positives divided by the number of true positives + false positives
    rec = torch.where(true_per_class > 0, tp / true_per_class, torch.zeros_like(tp)) #recall is the number of true positives divided by the number of true positives + false negatives
    f1 = torch.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), torch.zeros_like(tp)) #f1 score is the harmonic mean of precision and recall
    union = pred_per_class + true_per_class - tp #union is the number of true positives + false positives + false negatives
    iou = torch.where(union > 0, tp / union, torch.zeros_like(tp)) #iou is the number of true positives divided by the number of true positives + false positives + false negatives

    total = cm.sum()
    micro = (tp.sum() / total).item() if total > 0 else 0.0
    
    # overall metrics
    
    # note
    # micro: Treats every sample equally. 
    # better for gauging overall performance
    # macro: treat all classes equally 
    # better for spotting poor performance on rare classes
    out = {
        'precision_macro': prec.mean().item(),
        'recall_macro': rec.mean().item(),
        'f1_macro': f1.mean().item(),
        'mean_intersection_over_union': iou.mean().item(),
        'precision_micro': micro,
        'recall_micro': micro,
        'f1_micro': micro,
    }
    # per class metrics
    for c in range(num_classes):
        out[f'precision_c{c}'] = prec[c].item()
        out[f'recall_c{c}'] = rec[c].item()
        out[f'f1_c{c}'] = f1[c].item()
        out[f'iou_c{c}'] = iou[c].item()

        #debug for 4 class runs
        out[f'true_count_c{c}'] = true_per_class[c].item()
        out[f'pred_count_c{c}'] = pred_per_class[c].item()
        out[f'tp_c{c}'] = tp[c].item()

    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            out[f'cm_t{true_class}_p{pred_class}'] = cm[true_class, pred_class].item()
    return out

def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix='', output_dir='', epoch = None, vis_idx = None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    if vis_idx is None:
        vis_idx = []

    model.eval()

    patch_confusion = None
    if args.classifier_mode == 'patch':
        patch_confusion = torch.zeros(
            args.num_classes, args.num_classes, device='cuda', dtype=torch.float64
        )

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 3:
                input, target, image_paths = batch
            else:
                input, target = batch
                image_paths = None

            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)

            #Note: I took out the acc5 for now
            if args.classifier_mode == 'patch':
                # output: (B, num_classes, H, W)
                # target: (B, H, W)
                pred = output.argmax(dim=1)

                # mean patch accuracy over the batch
                #acc1 = (pred == target).float().mean() * 100.0
                # accuracy over valid (non-ignore) patches only
                valid_mask = (target != -100)
                if valid_mask.any():
                    acc1 = ((pred == target) & valid_mask).float().sum() / valid_mask.float().sum() * 100.0
                    
                    # update confusion matrix
                    valid_cells_flat = valid_mask.reshape(-1) #flatten the valid mask
                    pred_labeled = pred.reshape(-1)[valid_cells_flat] #flatten the predicted patches
                    target_labeled = target.reshape(-1)[valid_cells_flat] #flatten the target patches
                    n_cls = args.num_classes
                    confusion_linear_idx = target_labeled.long() * n_cls + pred_labeled.long() #linear index for the confusion matrix
                    patch_confusion.view(-1).index_add_(
                        0,
                        confusion_linear_idx, #index to add the ones to
                        torch.ones(pred_labeled.shape[0], device=patch_confusion.device, dtype=patch_confusion.dtype), #ones to add
                    )
                else:
                    acc1 = torch.tensor(0.0, device=target.device)

                # Save overlays for specific image names
                save_this_image = False
                image_name = None

                if image_paths is not None:
                    image_name = os.path.basename(image_paths[0])
                    image_stem = os.path.splitext(image_name)[0]
                    image_prefix = image_stem.split("_")[0]

                    # Change this list to whatever images you want saved
                    save_this_image = image_prefix in ["C1"]

                if args.local_rank == 0 and save_this_image:
                    vis_dir = os.path.join(output_dir if output_dir else "./output", "patch_vis", f"epoch_{epoch}")
                    os.makedirs(vis_dir, exist_ok=True)

                    # copy prediction map for overlay
                    pred_vis = pred[0].clone()

                    # no data regions appear red in prediction overlay
                    pred_vis[target[0] == -100] = -100

                    save_prediction_and_target_overlays(
                        input_tensor=input[0],
                        target_patch_map=target[0],
                        pred_patch_map=pred_vis,
                        save_path=os.path.join(vis_dir, f"val_overlay_epoch{epoch}_{image_prefix}.png"),
                        ignore_index=-100,
                        alpha=0.35,
                    )
            else:
                # image classification
                acc1 = accuracy(output, target, topk=(1,))[0]

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            #top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            #print logs for image vs patch classification
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                if args.classifier_mode == 'patch':
                    log_name = 'Test' + log_suffix
                    _logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'PatchAcc: {top1.val:>7.4f} ({top1.avg:>7.4f})'.format(
                            log_name, batch_idx, last_idx,
                            batch_time=batch_time_m,
                            loss=losses_m,
                            top1=top1_m
                        )
                    )
                else:
                    log_name = 'Test' + log_suffix
                    _logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                            log_name, batch_idx, last_idx,
                            batch_time=batch_time_m,
                            loss=losses_m,
                            top1=top1_m,
                            top5=top5_m
                        )
                    )

    #different metrics for image vs patch classification
    if args.classifier_mode == 'patch':
        metrics = OrderedDict([
            ('loss', losses_m.avg),
            ('patch_acc', top1_m.avg),
        ])
        if patch_confusion is not None:
            patch_classification_metrics = get_metrics_from_confusion_matrix(patch_confusion)
            for metric_name, value in patch_classification_metrics.items():
                metrics[metric_name] = value
    else:
        metrics = OrderedDict([
            ('loss', losses_m.avg),
            ('top1', top1_m.avg),
            #('top5', top5_m.avg),
        ])

    return metrics


if __name__ == '__main__':
    main()
