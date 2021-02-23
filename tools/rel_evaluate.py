# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import os
import sys
import argparse
import numpy as np
from os import mkdir
from tqdm import tqdm

import torch

sys.path.append('.')
from config import cfg
from modeling import build_model
from utils.logger import setup_logger
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from data.transforms import build_input_transforms
from sklearn.linear_model import LogisticRegression


def extract_features(model, dataloader):
    zs, labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader):
            batch_z = model.encode(batch_x.to(cfg.MODEL.DEVICE))
            zs.append(batch_z.cpu().numpy())
            labels.append(batch_y.numpy())
    zs = np.concatenate(zs, 0)
    labels = np.concatenate(labels)
    return zs, labels



def main():
    parser = argparse.ArgumentParser(description="PyTorch Template MNIST Inference")
    parser.add_argument(
        "--config-file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.OUTPUT_ROOT, cfg.PROJECT_NAME, cfg.EXPERIMENT_NAME)
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("template_model", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = build_model(cfg).eval().to(cfg.MODEL.DEVICE)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT)['model'])
    # load val data
    i_transform = build_input_transforms(cfg, is_train=False)
    train_dataset = ImageFolder(cfg.DATASETS.TRAIN_ROOT, i_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS
    )
    # load val data
    val_dataset = ImageFolder(cfg.DATASETS.TEST_ROOT, i_transform)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS
    )

    # get train zs
    train_zs, train_labels = extract_features(model, train_loader)
    val_zs, val_labels = extract_features(model, val_loader)

    logreg = LogisticRegression(solver='liblinear').fit(train_zs, train_labels)
    train_acc = logreg.score(train_zs, train_labels)
    val_acc = logreg.score(val_zs, val_labels)
    print('Train acc: {0:3.3f}, Val acc: {1:3.3f}'.format(train_acc, val_acc))


if __name__ == '__main__':
    main()
