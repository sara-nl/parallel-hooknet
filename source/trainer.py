from typing import Dict, Type
from random import randint
import sys
import time
import argparse
import random
import os
from abc import abstractmethod, ABC
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
import tensorflow.keras.backend as K
import time
from tqdm import tqdm
import pdb

from source.loss import HooknetLoss


class HookNetTrainer:
    """
    Trainer class specific for the HookNet model. It uses the validation loss to track the best model
    """

    def __init__(self,
                 model,
                 optimizer,
                 loss_weights,
                 compression,
                 logger,
                 log_every,
                 batchgenerator,
                 epochs,
                 steps_per_epoch_train,
                 validate_every,
                 batch_size,
                 output_path,
                 n_classes,
                 opts):

        self.epochs = epochs
        self.steps_per_epoch_train = steps_per_epoch_train
        self.validate_every = validate_every

        self.batchgenerator = batchgenerator
        self.model = model
        self.optimizer = optimizer
        self.compression = compression
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.opts = opts

        self.loss_func = HooknetLoss(loss_weights)
        self.logger = logger
        self.log_every = log_every
        self.best_metric = None
        self.weights_file = os.path.join(str(output_path), 'weights.h5')
        self.steps = 0

        self.accuracy_func = tf.keras.metrics.Accuracy()
        self.mio_func = tf.keras.metrics.MeanIoU(num_classes=n_classes)
        self.auc_func = tf.keras.metrics.AUC()
        self.recall_func = tf.keras.metrics.Recall()
        self.precision_func = tf.keras.metrics.Precision()

    def log_one_step(self, output, mode='Train'):

        if isinstance(output, dict):
            target_label, context_label = output['label'][0], output['label'][1]
            target_pred, context_pred = output['pred'][0], output['pred'][1]
            loss = output['loss']
        elif isinstance(output, list):
            target_label, context_label = np.array([x['label'][0] for x in output]), np.array([x['label'][1] for x in output])
            target_pred, context_pred = np.array([x['pred'][0] for x in output]), np.array([x['pred'][1] for x in output])
            loss = np.array([x['loss'] for x in output]).mean()
        else:
            raise NotImplemented(f'Not a valid type: {type(output)}')

        target_labels = hvd.allgather(target_label)
        context_labels = hvd.allgather(context_label)
        target_preds = hvd.allgather(target_pred)
        context_preds = hvd.allgather(context_pred)
        loss = hvd.allreduce(loss)

        if self.opts.horovod and hvd.local_rank() == 0 and hvd.rank() == 0:

            target_accuracy = self.accuracy_func(tf.math.argmax(target_labels, -1), tf.math.argmax(target_preds, -1))
            context_accuracy = self.accuracy_func(tf.math.argmax(context_labels, -1), tf.math.argmax(context_preds, -1))

            target_miou = self.mio_func(target_labels, target_preds)
            context_miou = self.mio_func(context_labels, context_preds)

            target_auc = self.auc_func(target_labels, target_preds)
            context_auc = self.auc_func(context_labels, context_preds)

            target_recall = self.recall_func(target_labels, target_preds)
            context_recall = self.recall_func(context_labels, context_preds)

            target_precision = self.precision_func(target_labels, target_preds)
            context_precision = self.precision_func(context_labels, context_preds)

            target_f1 = (2 * target_precision * target_recall) / (target_precision + target_recall + K.epsilon())
            context_f1 = (2 * context_precision * context_recall) / (context_precision + context_recall + K.epsilon())

            with self.logger.as_default():
                tf.summary.scalar(f'Loss/{mode}', loss, step=tf.cast(self.steps, tf.int64))

                tf.summary.scalar(f'Target accuracy/{mode}', target_accuracy, step=tf.cast(self.steps, tf.int64))
                tf.summary.scalar(f'Context accuracy/{mode}', context_accuracy, step=tf.cast(self.steps, tf.int64))

                tf.summary.scalar(f'Target mIoU/{mode}', target_miou, step=tf.cast(self.steps, tf.int64))
                tf.summary.scalar(f'Context mIoU/{mode}', context_miou, step=tf.cast(self.steps, tf.int64))

                tf.summary.scalar(f'Target AUC/{mode}', target_auc, step=tf.cast(self.steps, tf.int64))
                tf.summary.scalar(f'Context AUC/{mode}', context_auc, step=tf.cast(self.steps, tf.int64))

                tf.summary.scalar(f'Target Recall/{mode}', target_recall, step=tf.cast(self.steps, tf.int64))
                tf.summary.scalar(f'Context Recall/{mode}', context_recall, step=tf.cast(self.steps, tf.int64))

                tf.summary.scalar(f'Target Precision/{mode}', target_precision, step=tf.cast(self.steps, tf.int64))
                tf.summary.scalar(f'Context Precision/{mode}', context_precision, step=tf.cast(self.steps, tf.int64))

                tf.summary.scalar(f'Target F1/{mode}', target_f1, step=tf.cast(self.steps, tf.int64))
                tf.summary.scalar(f'Context F1/{mode}', context_f1, step=tf.cast(self.steps, tf.int64))

            self.logger.flush()

    def train(self):
        """Train loop"""
        print("Start training")
        for epoch in range(self.epochs):
            self.epoch()

    def epoch(self):

        ### TRAINING PART ###
        epoch_metrics_train = []

        if (self.opts.horovod and hvd.local_rank() == 0 and hvd.rank() == 0) or not self.opts.horovod:
            pbar = tqdm(total=self.steps_per_epoch_train)

        for _ in range(self.steps_per_epoch_train):
            outputs = self.train_one_step()
            epoch_metrics_train.append(outputs)

            if self.steps % self.log_every == 0 and self.steps > 0:
                self.log_one_step(outputs)

            if self.steps % self.validate_every == 0:
                outputs = self.val_one_step()
                self.log_one_step(outputs, mode='Validation')

            if (self.opts.horovod and hvd.local_rank() == 0 and hvd.rank() == 0) or not self.opts.horovod:
                pbar.update(1)

        self.on_train_epoch_end(epoch_metrics_train)

        # ### VALIDATION PART ###
        # epoch_metrics_val = []
        #
        # if hvd.local_rank() == 0 and hvd.rank() == 0:
        #     pbar = tqdm(total=self.steps_per_epoch_val)
        #
        # for _ in range(self.steps_per_epoch_val):
        #     outputs = self.val_one_step()
        #     epoch_metrics_val.append(outputs)
        #
        #     if hvd.local_rank() == 0 and hvd.rank() == 0:
        #         pbar.update(1)
        #
        # self.on_val_epoch_end(epoch_metrics_val)
        # self.batch_generator_validation.reset('validation')

    def on_train_epoch_end(self, epoch_metrics):
        """ Things to perform on finishing the training epoch """
        pass

    def on_val_epoch_end(self, epoch_metrics):
        """ Things to perform on finishing a validation epoch """
        self.log_one_step(epoch_metrics, mode='Val')

    def train_one_step(self):
        """Step method that retrieves a batch based on the set batch function and applies the model based on the set model function"""
        start_time = time.time()

        batch = self.batchgenerator.batch('training')

        batch_gen_time = time.time()

        x_batch, y_batch = batch['x_batch'], batch['y_batch']
        x_batch = [x_batch[:, i] for i in range(x_batch.shape[1])]
        y_batch = [y_batch[:, i] for i in range(y_batch.shape[1])]

        with tf.GradientTape() as tape:
            probabilities = self.model(x_batch)
            loss = self.loss_func(probabilities, y_batch)

        if self.opts.horovod:
            tape = hvd.DistributedGradientTape(tape, compression=self.compression)
        grads = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.opts.horovod and self.steps == 0:
            hvd.broadcast_variables(self.model.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        self.steps += 1

        if self.opts.horovod and hvd.local_rank() == 0 and hvd.rank() == 0:
            print(f'Batch generator took {batch_gen_time - start_time}')

        return {'loss': loss,
                'pred': probabilities,
                'label': y_batch}

    def val_one_step(self):
        """Step method that retrieves a batch based on the set batch function and applies the model based on the set model function"""
        batch = self.batchgenerator.batch('validation')
        x_batch, y_batch = batch['x_batch'], batch['y_batch']
        x_batch = [x_batch[:, i] for i in range(x_batch.shape[1])]
        y_batch = [y_batch[:, i] for i in range(y_batch.shape[1])]

        probabilities = self.model(x_batch)
        loss = self.loss_func(probabilities, y_batch)

        return {'loss': loss,
                'pred': probabilities,
                'label': y_batch}
