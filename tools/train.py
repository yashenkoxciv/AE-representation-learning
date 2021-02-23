# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import makedirs

import torch.nn.functional as F

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer

from utils.logger import setup_logger


def train(cfg):
    model = build_model(cfg).to(cfg.MODEL.DEVICE)
    device = cfg.MODEL.DEVICE

    optimizer = make_optimizer(cfg, model)
    scheduler = None

    arguments = {}

    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        None,
        F.mse_loss,
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Template MNIST Training")
    parser.add_argument(
        "--config-file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.OUTPUT_ROOT, cfg.PROJECT_NAME, cfg.EXPERIMENT_NAME)
    makedirs(output_dir, exist_ok=True)

    logger = setup_logger(cfg.EXPERIMENT_NAME, output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg)


if __name__ == '__main__':
    main()
