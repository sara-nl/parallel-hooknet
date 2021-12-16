import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Add, BatchNormalization, Conv2D, Subtract, Multiply, Reshape, MaxPooling2D, \
    UpSampling2D, Cropping2D, concatenate
import horovod.tensorflow as hvd


def hooknet(input_shape,
            n_classes,
            hook_indexes,
            depth=4,
            n_convs=2,
            filter_size=3,
            n_filters=64,
            padding='valid',
            batch_norm=True,
            activation='relu',
            learning_rate=0.000005,
            l2_lambda=0.001,
            loss_weights=[1.0, 0.0],
            merge_type='concat',
            horovod=False,
            fp16_allreduce=False):

    hook_indexes = {depth - hook_indexes[0]: hook_indexes[1]}

    # set l2 regulizer
    l2 = regularizers.l2(l2_lambda)

    # construct model
    input_1 = Input(input_shape)
    input_2 = Input(input_shape)

    flatten2, context_hooks = construct_branch(input_2, {}, 'context', n_classes, n_filters, depth, n_convs,
                                               activation, padding, l2, batch_norm, merge_type, hook_indexes,
                                               filter_size)

    # construction of target branch with context hooks
    flatten1, _ = construct_branch(input_1, context_hooks, 'target', n_classes, n_filters, depth, n_convs,
                                   activation, padding, l2, batch_norm, merge_type, hook_indexes, filter_size)

    # create multi loss model
    model = Model([input_1, input_2], [flatten1, flatten2], name='hooknet')

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if horovod:
        compression = hvd.Compression.fp16 if fp16_allreduce else hvd.Compression.none
    else:
        compression = None

    # hooknet_loss = HooknetLoss(loss_weights)
    # metrics = [tf.keras.metrics.Accuracy(),
    #            tf.keras.metrics.MeanIoU(num_classes=n_classes),
    #            tf.keras.metrics.AUC(),
    #            tf.keras.metrics.Recall(),
    #            tf.keras.metrics.Precision(),
    #            f1_score]
    #
    # model.compile(optimizer=optimizer, loss=hooknet_loss, metrics=metrics)

    return model, optimizer, compression


def construct_branch(input,
                     in_hooks,
                     reshape_name,
                     n_classes,
                     n_filters,
                     depth,
                     n_convs,
                     activation,
                     padding,
                     l2,
                     batch_norm,
                     merge_type,
                     hook_indexes,
                     filter_size):
    # input
    net = input

    # encode and retreive residuals
    net, residuals = encode_path(net, n_filters, depth, n_convs, activation, padding, l2, batch_norm)

    # mid conv block
    net = conv_block(net, n_filters * 2 * (depth + 1), n_convs, activation, padding, l2, batch_norm)

    # decode and retreive hooks
    net, out_hooks = decode_path(net, residuals, in_hooks, n_filters, depth, merge_type, hook_indexes, activation,
                                 padding, filter_size, l2, n_convs, batch_norm)

    # softmax output
    net = Conv2D(n_classes, 1, activation='softmax')(net)

    # set output shape
    output_shape = tf.keras.backend.int_shape(net)[1:]

    # Reshape net
    flatten = Reshape((output_shape[0], output_shape[1], output_shape[2]), name=reshape_name)(net)

    # return flatten output and hooks
    return flatten, out_hooks


def encode_path(net, n_filters, depth, n_convs, activation, padding, l2, batch_norm):
    # list for keeping track for residuals/skip connections
    residuals = []

    # set start filtersize
    n_filters = n_filters

    # loop through depths
    for b in range(depth):
        # apply convblock
        net = conv_block(net,
                         n_filters,
                         n_convs,
                         activation,
                         padding,
                         l2,
                         batch_norm)

        # keep Tensor for residual/sip connection
        residuals.append(net)

        # downsample
        net = downsample(net)

        # increase number of filters with factor 2
        n_filters *= 2

    return net, residuals


def decode_path(net,
                residuals,
                inhooks,
                n_filters,
                depth,
                merge_type,
                hook_indexes,
                activation,
                padding,
                filter_size,
                l2,
                n_convs,
                batch_norm):
    # list for keeping potential hook Tensors
    outhooks = []

    # set start number of filters of decoder
    n_filters = n_filters * 2 * depth

    # loop through depth in reverse
    for b in reversed(range(depth)):

        # hook if hook is available
        if b in inhooks:
            # combine feature maps via merge type
            if merge_type == 'concat':
                net = concatenator(net, inhooks[b])
            else:
                net = merger(net, inhooks[b], activation, padding, merge_type)

        # upsample
        net = upsample(net, n_filters, filter_size, activation, padding, l2)

        # concatenate residuals/skip connections
        net = concatenator(net, residuals[b])

        # apply conv block
        net = conv_block(net, n_filters, n_convs, activation, padding, l2, batch_norm)

        # set potential hook
        outhooks.append(net)

        n_filters = n_filters // 2

    # get hooks from potential hooks
    hooks = {}
    for shook, ehook in hook_indexes.items():
        hooks[ehook] = outhooks[shook]

    print(type(net))
    return net, hooks


def conv_block(net,
               n_filters,
               n_convs,
               activation,
               padding,
               l2,
               batch_norm,
               kernel_size=3):
    # loop through number of convolutions in convolution block
    for n in range(n_convs):
        # apply 2D convolution
        net = Conv2D(n_filters,
                     kernel_size,
                     activation=activation,
                     kernel_initializer='he_normal',
                     padding=padding,
                     kernel_regularizer=l2)(net)

        # apply batch normalization
        if batch_norm:
            net = BatchNormalization()(net)

    return net


def downsample(net):
    """Downsampling via max pooling"""

    return MaxPooling2D(pool_size=(2, 2))(net)


def upsample(net, n_filters, filter_size, activation, padding, l2):
    """Upsamplign via nearest neightbour interpolation and additional convolution"""

    net = UpSampling2D(size=(2, 2))(net)
    net = Conv2D(n_filters,
                 filter_size,
                 activation=activation,
                 padding=padding,
                 kernel_regularizer=l2)(net)

    return net


def concatenator(net, item):
    """"Concatenate feature maps"""

    # crop feature maps
    crop_size = int(item.shape[1] - net.shape[1]) / 2
    item_cropped = Cropping2D(int(crop_size))(item)

    return concatenate([item_cropped, net], axis=3)


def merger(net, item, activation, padding, merge_type):
    """"Combine feature maps"""

    # crop feature maps
    crop_size = int(item.shape[1] - net.shape[1]) / 2
    item_cropped = Cropping2D(int(crop_size))(item)

    # adapt number of filters via 1x1 convolutional to allow merge
    current_filters = int(net.shape[-1])
    item_cropped = Conv2D(current_filters,
                          1,
                          activation=activation,
                          padding=padding)(item_cropped)

    # Combine feature maps by adding
    if merge_type == 'add':
        return Add()([item_cropped, net])
    # Combine feature maps by subtracting
    if merge_type == 'subtract':
        return Subtract()([item_cropped, net])
    # Combine feature maps by multiplication
    if merge_type == 'multiply':
        return Multiply()([item_cropped, net])

    # Raise ValueError if merge type is unsupported
    raise ValueError(f'unsupported merge type: {merge_type}')


def create_model(inputs, outputs):
    return Model(inputs, outputs, name='hooknet')
