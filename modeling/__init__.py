# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline_ae import ResNet18AE


def build_model(cfg):
    model = ResNet18AE(z_dim=1024, pretrained_backbone=True, fine_tuning=True)
    return model
