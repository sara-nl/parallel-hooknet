from tensorflow.keras import backend as K
import numpy as np
import os
import tensorflow.keras.backend as K
from tensorflow.python.ops import summary_ops_v2
import horovod.tensorflow as hvd
import tensorflow as tf
import random
import math


class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, opts):
        super(SaveModelCallback, self).__init__()
        self.opts = opts

    def on_epoch_end(self, epoch, logs={}):
        if epoch not in self.opts.epochs_to_save:
            return

        if not self.opts.horovod or (hvd.rank() == 0 and hvd.local_rank() == 0):
            self.model.save_weights(os.path.join(self.opts.output_dir, "tr-" + str(epoch) + ".h5"), save_format="h5")


class TensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, *args, **kwargs):
        super(TensorBoard, self).__init__(*args, **kwargs)

    def _log_epoch_metrics(self, epoch, logs):
        """Writes epoch metrics out as scalar summaries.
        Arguments:
            epoch: Int. The global step to use for TensorBoard.
            logs: Dict. Keys are scalar summary names, values are scalars.
        """
        if not logs:
            return

        train_logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        val_logs = {k: v for k, v in logs.items() if k.startswith('val_')}

        with summary_ops_v2.always_record_summaries():
            if train_logs:
                with self._train_writer.as_default():
                    for name, value in train_logs.items():
                        summary_ops_v2.scalar('Train/' + name, value, step=epoch)
            if val_logs:
                with self._train_writer.as_default():
                    for name, value in val_logs.items():
                        name = name[4:]  # Remove 'val_' prefix.
                        summary_ops_v2.scalar('Val/' + name, value, step=epoch)
