from source.inference import Inference
from source.model import HookNet
import glob
import os
import sys
from argconfigparser import ArgumentConfigParser
import os.path
import shutil
from pprint import pprint
import json
import pdb


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


# number of cpus for wsireader deamon
cpus = 6

# queue size for sending batches
queue_size = 10

# parse arguments
parser = ArgumentConfigParser('./parameters.yml', description='HookNet')
parser.add_argument("-i", '--image_path', dest="image_path", required=True,
                    help="input image path", metavar="FILE_PATH", type=lambda x: is_valid_file(parser, x))
parser.add_argument("-m", '--mask_path', dest="mask_path", required=False,
                    help="mask image path", metavar="FILE_PATH", type=lambda x: is_valid_file(parser, x))
parser.add_argument("-w", '--weights_path', dest="weights_path", required=True,
                    help="weights file path", metavar="FILE_PATH", type=lambda x: is_valid_file(parser, x))
parser.add_argument("-d", '--work_dir', dest="work_dir", required=True,
                    help="work directory", metavar="FILE_PATH", type=lambda x: is_valid_file(parser, x))
config = parser.parse_args()
pprint(f'CONFIG: \n{config}')


with open(os.path.join(config.work_dir, "args.json"), encoding='utf-8', errors='ignore') as json_data:
    data = json.load(json_data, strict=False)
    opts = Namespace(**data)

# initialize model
hooknet, _, _ = HookNet(input_shape=opts.input_shape,
                        n_classes=opts.n_classes,
                        hook_indexes=opts.hook_indexes,
                        depth=opts.depth,
                        n_convs=opts.n_convs,
                        filter_size=opts.filter_size,
                        n_filters=opts.n_filters,
                        padding=opts.padding,
                        batch_norm=opts.batch_norm,
                        activation=opts.activation,
                        learning_rate=opts.learning_rate,
                        l2_lambda=opts.l2_lambda,
                        loss_weights=opts.loss_weights,
                        merge_type=opts.merge_type,
                        horovod=opts.horovod,
                        fp16_allreduce=opts.fp16_allreduce)

# load weights
hooknet.load_weights(config['weights_path'])

true_image_path = config['image_path']
true_mask_path = config['mask_path']
true_output_path = config['output_path']

print('image_path:', true_image_path)
print('mask_path:', true_mask_path)
print('output_path:', true_output_path)

pdb.set_trace()

print('copy to local folder...')
shutil.copy2(true_image_path, config['work_dir'])
if true_mask_path:
    shutil.copy2(true_mask_path, config['work_dir'])

image_path = os.path.join(config['work_dir'], os.path.basename(true_image_path))
mask_path = os.path.join(config['work_dir'], os.path.basename(true_mask_path)) if true_mask_path else true_mask_path
output_path = os.path.join(config['work_dir'], os.path.basename(true_image_path).replace('.', '_hooknet.'))

print('apply hooknet...')
apply = Inference(image_path,
                  mask_path,
                  output_path,
                  config['input_shape'],
                  hooknet.output_shape,
                  config['resolutions'],
                  config['batch_size'],
                  cpus,
                  queue_size,
                  hooknet)
apply.start()

print('copy result...')
shutil.copy2(output_path, true_output_path)
print('done.')
