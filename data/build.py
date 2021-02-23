# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from torch.utils import data

from .transforms import build_input_transforms, build_reconstruction_transforms
from .datasets.custom import GenerativeImageFolder


def build_dataset(cfg, input_transform, reconstruction_transform, is_train=True):
    dataset = GenerativeImageFolder(
        cfg.DATASETS.TRAIN_ROOT,
        input_transform=input_transform,
        reconstruction_transform=reconstruction_transform
    )
    return dataset


def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False

    input_transforms = build_input_transforms(cfg, is_train)
    reconstruction_transforms = build_reconstruction_transforms(cfg, is_train)
    datasets = build_dataset(cfg, input_transforms, reconstruction_transforms, is_train)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader
