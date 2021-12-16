import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import horovod.tensorflow as hvd
import random
from tensorboardX import SummaryWriter
from varname import nameof
import pdb


def init_horovod(opts):
    """ Run initialisation options"""
    if opts.horovod:
        hvd.init()
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        if opts.cuda:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print("hvd.size() = ", hvd.size())
            print("GPU's", gpus, "with Local Rank", hvd.local_rank())
            print("GPU's", gpus, "with Rank", hvd.rank())

            if gpus:
                tf.config.experimental.set_visible_devices(gpus[hvd.local_rank() % 4], 'GPU')

        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        opts.seed += hvd.rank()
        tf.random.set_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)

        opts.learning_rate *= hvd.size()
    else:
        tf.random.set_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)


def setup_logger(opts):
    """ Setup the tensorboard writer """
    # Sets up a timestamped log directory.
    logdir = opts.output_dir
    os.makedirs(logdir, exist_ok=True)

    if opts.horovod:
        # Creates a file writer for the log directory.
        if hvd.local_rank() == 0 and hvd.rank() == 0:
            file_writer = SummaryWriter(logdir, flush_secs=1)
        else:
            file_writer = None
    else:
        # If running without horovod
        file_writer = SummaryWriter(logdir, flush_secs=1)

    return file_writer


def get_output_size(model, opts):
    """ Run a batch through the model to determine the output size """
    dim = opts.input_shape[0]

    for _ in range(opts.depth):
        dim = (dim - 4) / 2

    dim -= 4

    for _ in range(opts.depth):
        dim = (dim - 3) * 2

    return [int(dim), int(dim), opts.n_classes]


def data_generator(batch_generator, mode):
    """ Pack the batch_generator object from Mart into a python generator object"""
    while True:
        yield batch_generator.batch(mode)


def f1_score_func(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1, precision, recall


def accuracy_func(y_true, y_pred):
    y_true = np.argmax(y_true, -1)
    y_pred = np.argmax(y_pred, -1)

    correct = y_true == y_pred
    accuracy = np.sum(correct) / np.prod(correct.shape)

    return accuracy


def calc_multi_cls_measures(y_true, y_pred, opts):
    metrics = {}

    target_true, context_true = y_true
    target_pred, context_pred = y_pred

    target_pred = target_pred.numpy()
    context_pred = context_pred.numpy()

    miou_func = tf.keras.metrics.MeanIoU(num_classes=opts.n_classes)
    auc_func = tf.keras.metrics.AUC()

    target_f1, target_precision, target_recall = f1_score_func(target_true, target_pred)
    target_miou = miou_func(target_true, target_pred)
    target_auc = auc_func(target_true, target_pred)
    target_accuracy = accuracy_func(target_true, target_pred)

    context_f1, context_precision, context_recall = f1_score_func(context_true, context_pred)
    context_miou = miou_func(context_true, context_pred)
    context_auc = auc_func(context_true, context_pred)
    context_accuracy = accuracy_func(context_true, context_pred)

    metrics[nameof(target_f1)] = target_f1.numpy()
    metrics[nameof(target_precision)] = target_precision.numpy()
    metrics[nameof(target_recall)] = target_recall.numpy()
    metrics[nameof(target_miou)] = target_miou.numpy()
    metrics[nameof(target_auc)] = target_auc.numpy()
    metrics[nameof(target_accuracy)] = target_accuracy

    metrics[nameof(context_f1)] = context_f1.numpy()
    metrics[nameof(context_precision)] = context_precision.numpy()
    metrics[nameof(context_recall)] = context_recall.numpy()
    metrics[nameof(context_miou)] = context_miou.numpy()
    metrics[nameof(context_auc)] = context_auc.numpy()
    metrics[nameof(context_accuracy)] = context_accuracy

    return metrics


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def log_one_step(logger, logs, postfix, step):
    for key, value in logs.items():
        logger.add_scalar(key + postfix, value, step)


def save_model(model, opts):
    if (opts.horovod and hvd.rank() == 0) or not opts.horovod:
        model.save_weights(os.path.join(opts.output_dir, 'model.h5'), overwrite=True, save_format='h5')


def correct_size(x):
    if (x - 4) % 2 == 0:
        x = x - 4
        if (x - 8) % 2 == 0:
            x = (x - 8) / 2
            if (x - 8) % 2 == 0:
                x = (x - 8) / 2
                if (x - 8) % 2 == 0:
                    x = (x - 8) / 2
                    if (x - 8) % 2 == 0:
                        x = (x - 8) / 2
                        if x % 2 == 0:
                            return True
    return False


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--depth', type=int, default=4)
    opts = p.parse_args()
    opts.input_shape = [284, 284, 3]
    opts.n_classes = 2

    model = None
    output_size = get_output_size(model, opts)
    print()
