# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Loss
import torch.nn.functional as F


def inference(
        cfg,
        model,
        val_loader
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("template_model.inference")
    logger.info("Start inferencing")
    evaluator = create_supervised_evaluator(
        model, metrics={
            'loss': Loss(F.mse_loss)
        },
        device=device)

    # adding handlers using `evaluator.on` decorator API
    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_validation_results(engine):
        metrics = evaluator.state.metrics
        avg_loss = metrics['loss']
        logger.info("Validation Results - Loss: {:.3f}".format(avg_loss))

    evaluator.run(val_loader)
