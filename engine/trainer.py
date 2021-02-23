# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import os
import logging

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Loss, RunningAverage


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
):
    log_period = cfg.SOLVER.LOG_PERIOD
    #checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    output_dir = os.path.join(cfg.OUTPUT_ROOT, cfg.PROJECT_NAME, cfg.EXPERIMENT_NAME)

    logger = logging.getLogger('-'.join([cfg.PROJECT_NAME, cfg.EXPERIMENT_NAME]))
    logging.basicConfig(level=logging.INFO)
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(
        model, metrics={
            #'accuracy': Accuracy(),
            'ae_loss': Loss(loss_fn)
        }, device=device
    )
    checkpointer = ModelCheckpoint(output_dir, cfg.EXPERIMENT_NAME, None, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, checkpointer,
        {
            'model': model,
            'optimizer': optimizer
        }
    )
    timer.attach(
        trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED
    )

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                        .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_loss = metrics['ae_loss']
        logger.info("Training Results - Epoch: {} Avg Loss: {:.3f}"
                    .format(engine.state.epoch, avg_loss))

    if val_loader is not None:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            avg_loss = metrics['ae_loss']
            logger.info("Validation Results - Epoch: {} Avg Loss: {:.3f}"
                        .format(engine.state.epoch, avg_loss)
                        )

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        timer.reset()

    trainer.run(train_loader, max_epochs=epochs)
