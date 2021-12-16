import os
import copy
import math
from tqdm import tqdm
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import horovod.tensorflow as hvd
import time
import pdb

from utils import data_generator, calc_multi_cls_measures, dict_mean, log_one_step, save_model
from source.loss import HooknetLoss
from callbacks import SaveModelCallback, TensorBoard


def train(model, optimizer, batch_generator, logger, opts):
    train_generator = data_generator(batch_generator, 'training')
    val_generator = data_generator(copy.deepcopy(batch_generator), 'validation')

    if opts.horovod:
        steps_per_epoch_train = int(math.ceil(opts.steps_per_epoch_train) / hvd.size())
        steps_per_epoch_val = int(math.ceil(opts.val_steps) / hvd.size())
    else:
        steps_per_epoch_train = opts.steps_per_epoch_train
        steps_per_epoch_val = opts.val_steps

    if opts.debug:
        steps_per_epoch_train, steps_per_epoch_val = 10, 10

    loss = HooknetLoss(opts.loss_weights)

    for epoch in tqdm(range(opts.epochs)):

        # Perform the training
        train_metrics = []
        for step in tqdm(range(steps_per_epoch_train)):
            train_metric = train_one_step(model, optimizer, loss, train_generator, epoch, step, opts)
            train_metrics.append(train_metric)

        if opts.horovod:
            train_metrics = hvd.allgather_object(train_metrics)
            train_metrics = [item for sublist in train_metrics for item in sublist]
        train_metrics = dict_mean(train_metrics)

        # Perform the logging step every 'log_every' epochs
        if (epoch % opts.validate_every) == 0 or (epoch == opts.epochs - 1):
            val_metrics = []
            for _ in range(steps_per_epoch_val):
                val_metric = val_one_step(model, loss, val_generator, opts)
                val_metrics.append(val_metric)

            if opts.horovod:
                val_metrics = hvd.allgather_object(val_metrics)
                val_metrics = [item for sublist in val_metrics for item in sublist]
            val_metrics = dict_mean(val_metrics)
        else:
            val_metrics = {}

        # Log to tensorboard
        if (opts.horovod and hvd.rank() == 0) or not opts.horovod:
            log_one_step(logger, train_metrics, '/Train', epoch)
            log_one_step(logger, val_metrics, '/Val', epoch)

    save_model(hooknet, opts)


def train_one_step(model, optimizer, loss, train_generator, epoch, step, opts):
    x_batch, y_batch = next(train_generator)
    with tf.GradientTape() as tape:
        probs = model(x_batch, training=True)
        loss_value = loss(probs, y_batch)

    if opts.horovod:
        tape = hvd.DistributedGradientTape(tape)

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if opts.horovod and epoch == step == 0:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(optimizer.variables(), root_rank=0)

    train_metrics = calc_multi_cls_measures(y_batch, probs, opts)
    train_metrics['loss'] = loss_value.numpy()
    train_metrics['rank'] = hvd.rank()

    return train_metrics


def val_one_step(model, loss, val_generator, opts):
    x_batch, y_batch = next(val_generator)

    probs = model(x_batch, training=True)
    loss_value = loss(probs, y_batch)

    val_metrics = calc_multi_cls_measures(y_batch, probs, opts)
    val_metrics['loss'] = loss_value.numpy()
    val_metrics['rank'] = hvd.rank()

    return val_metrics
