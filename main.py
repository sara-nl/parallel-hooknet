import os
from pprint import pprint
import json
import pdb

from source.model_tf_2 import hooknet as HookNet
from train import train
from xmlpathology.batchgenerator.utils import create_data_source
from xmlpathology.batchgenerator.generators import BatchGenerator
from xmlpathology.batchgenerator.core.samplers import LabelSamplerLoader, SamplerLoader
from xmlpathology.batchgenerator.core.samplers import SegmentationLabelSampler, Sampler
from xmlpathology.batchgenerator.callbacks import OneHotEncoding, FitData
from utils import init_horovod, get_output_size, setup_logger, save_model
from options import get_options


def main(opts):
    init_horovod(opts)

    # initialize model
    hooknet, optimizer, compression = HookNet(input_shape=opts.input_shape,
                                              n_classes=opts.n_classes,
                                              hook_indexes=opts.hook_indexes,
                                              depth=opts.depth,
                                              n_convs=opts.n_convs,
                                              filter_size=opts.filter_size,
                                              n_filters=opts.n_filters,
                                              padding=opts.padding,
                                              #batch_norm=opts.batch_norm,
                                              batch_norm=False,
                                              activation=opts.activation,
                                              learning_rate=opts.learning_rate,
                                              l2_lambda=opts.l2_lambda,
                                              loss_weights=opts.loss_weights,
                                              merge_type=opts.merge_type,
                                              horovod=opts.horovod,
                                              fp16_allreduce=opts.fp16_allreduce)

    hooknet.summary()

    opts.output_shape = get_output_size(hooknet, opts)

    pprint(vars(opts))
    with open(os.path.join(opts.output_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Create the datasource
    datasource = create_data_source(data_folder=opts.train_data_path,
                                    annotations_path=opts.train_annotations_path,
                                    images_extension='.tif',
                                    annotations_extension='.xml',
                                    mode='training')

    print(f'Found {len(datasource["training"])} training WSIs')

    datasource_valid = create_data_source(data_folder=opts.valid_data_path,
                                          annotations_path=opts.valid_annotations_path,
                                          images_extension='.tif',
                                          annotations_extension='.xml',
                                          mode='validation')

    print(f'Found {len(datasource_valid["validation"])} validation WSIs')

    datasource = {**datasource, **datasource_valid}

    # label_map = {'_0': 1, '_2': 2}
    label_map = {"dcis": 1,
                 "idc": 2,
                 "ilc": 3,
                 "stroma": 4,
                 "fatty tissue": 5,
                 "inflammatory cells": 4,
                 "skin/nipple": 4,
                 "erythrocytes": 4,
                 "non malignant epithelium": 6}

    # initialize batchgenerator
    batch_generator = BatchGenerator(data_sources=datasource,
                                     label_map=label_map,
                                     batch_size=opts.batch_size,
                                     sampler_loader=SamplerLoader(class_=Sampler,
                                                                  input_shapes=[opts.input_shape],
                                                                  spacings=opts.resolutions),
                                     label_sampler_loader=LabelSamplerLoader(class_=SegmentationLabelSampler),
                                     log_path=opts.output_dir,
                                     cpus=3,
                                     seed=opts.seed,
                                     sample_callbacks=[FitData(opts.output_shape), OneHotEncoding()])

    logger = setup_logger(opts)
    save_model(hooknet, opts)

    train(hooknet, optimizer, batch_generator, logger, opts)
    logger.flush()


if __name__ == "__main__":
    opts = get_options()
    main(opts)
