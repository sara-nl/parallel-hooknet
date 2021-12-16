import argparse
import time
import os


def get_options():
    parser = argparse.ArgumentParser(description="Argument parser for parallel hooknet implementation")

    # Data options
    parser.add_argument('--train_data_path', type=str, default='/nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor',
                        help='Path to data folder')
    parser.add_argument('--train_annotations_path', type=str, help='Path to annotations folder',
                        default='/nfs/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/XML')
    parser.add_argument('--valid_data_path', type=str, default='/nfs/managed_datasets/CAMELYON16/Testset/Images',
                        help='Path to data folder')
    parser.add_argument('--valid_annotations_path', type=str, help='Path to annotations folder',
                        default='/nfs/managed_datasets/CAMELYON16/Testset/Ground_Truth/Annotations')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='Path where outputs (log files and weights) will be saved')
    parser.add_argument('--run_name', type=str, default='run', help='Name of the run')
    parser.add_argument('--validate_every', type=int, default=1, help='Perform a validation step every N epochs')

    # Model options
    parser.add_argument('--input_shape', type=int, default=284,
                        help='H or W of the size of the input images. Only square images supported for now')
    parser.add_argument('--n_classes', type=int, default=6, help='Number of classification classes')
    parser.add_argument('--resolutions', nargs='+', default=[0.5, 8.0], type=float,
                        help='Input resolutions of the model [target, context]')
    parser.add_argument('--hook_indexes', nargs='+', default=[3, 3], type=int,
                        help='The respective depths (starting from 0) of hooking [from, to] in the decoders')
    parser.add_argument('--n_convs', type=int, default=2, help='The number of 2D convolutions per convolutional block')
    parser.add_argument('--depth', type=int, default=4, help='The depth of the encoder-decoder branches')
    parser.add_argument('--n_filters', type=int, default=64,
                        help='The number of starting filters (will be increased and decreased by a factor 2 in each conv block in the encoders and decoders, respectively)')
    parser.add_argument('--filter_size', type=int, default=3, help='The size of the filter in a 2D convolution')
    parser.add_argument('--padding', type=str, default='valid',
                        help="Padding type in 2D convolution (either 'same' or 'valid')")
    parser.add_argument('--no_batch_norm', action='store_true', help='Boolean for not using batch normalization')
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function applied after 2D convolution')
    parser.add_argument('--learning_rate', type=float, default=0.000005, help='Learning rate of the optimizer')
    parser.add_argument('--l2_lambda', type=float, default=0.0001, help='L2 value for regulizer ')
    parser.add_argument('--opt_name', type=str, default='adam', choices=['adam', 'sgd'],
                        help="Optimizer name (either 'sgd' or 'adam')")
    parser.add_argument('--loss_weights', nargs='+', type=float, default=[0.5, 0.5],
                        help='Loss contribution for each branch [target, context]')
    parser.add_argument('--merge_type', type=str, default='concat',
                        help="Method used for combining feature maps (either 'concat', 'add', 'subtract', 'multiply')",
                        choices=['concat', 'add', 'substract', 'multiply'])

    # Train options
    parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs the trainer will run')
    parser.add_argument('--steps_per_epoch_train', type=int, default=200,
                        help='The number of steps (i.e., batches) in a training epoch')
    parser.add_argument('--val_steps', type=int, default=25, help='Run this many validation batches')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of examples in one training batch')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='The number of examples in one validation batch')
    parser.add_argument('--debug', action='store_true', help='Run in debugging mode')

    # System options
    parser.add_argument('--horovod', action='store_true', help='Run with horovod or not', default=False)
    parser.add_argument('--no_cuda', action='store_true', help='Run on CUDA or not', default=False)
    parser.add_argument('--fp16_allreduce', action='store_true', help='Use FP16 precision')
    parser.add_argument('--seed', type=int, default=0, help='Seed value for random python and numpy parts')
    parser.add_argument('--epochs_to_save', nargs='+', type=int, default=[197, 198, 199],
                        help='When to save model')

    opts = parser.parse_args()

    opts.input_shape = [opts.input_shape, opts.input_shape, 3]
    opts.batch_norm = not opts.no_batch_norm
    opts.cuda = not opts.no_cuda

    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.output_dir = os.path.join(opts.output_path, opts.run_name)
    os.makedirs(opts.output_dir, exist_ok=True)

    return opts
