# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torchvision.transforms as T

#from .transforms import RandomErasing


def build_input_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        input_transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            # T.RandomResizedCrop(
            #     size=cfg.INPUT.SIZE_TRAIN,
            #     scale=(cfg.INPUT.MIN_SCALE_TRAIN, cfg.INPUT.MAX_SCALE_TRAIN)
            # ),
            # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.ToTensor(),
            normalize_transform,
            #RandomErasing(probability=cfg.INPUT.PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        input_transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform
        ])
    return input_transform


def build_reconstruction_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        reconstruction_transform = T.Compose([
            T.Resize(cfg.RECONSTRUCTION.SIZE_TRAIN),
            # T.RandomResizedCrop(
            #     size=cfg.INPUT.SIZE_TRAIN,
            #     scale=(cfg.INPUT.MIN_SCALE_TRAIN, cfg.INPUT.MAX_SCALE_TRAIN)
            # ),
            # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.ToTensor(),
            normalize_transform,
            #RandomErasing(probability=cfg.INPUT.PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        reconstruction_transform = T.Compose([
            T.Resize(cfg.RECONSTRUCTION.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform
        ])
    return reconstruction_transform

