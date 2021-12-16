from typing import List, Dict, Tuple

import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Cropping2D, concatenate, Add, Subtract, \
    Multiply, Input, Reshape, Lambda, Concatenate
from keras.models import Model, K
from keras.optimizers import Optimizer, SGD, Adam

import pdb


class HookNet(Model):
    """
    HookNet: a convolutional neural network with mulitple branches of encoder-decoders for the task of semantic segmenation.

    ...

    Attributes
    ----------

    input_shape: List[int]
        the input shape of the model 

    output_shape: List[int]
        the output shape of the model before flattening. 

    """

    def __init__(self,
                 input_shape: List[int],
                 n_classes: int,
                 hook_indexes: List[int],
                 depth: int = 4,
                 n_convs: int = 2,
                 filter_size: int = 3,
                 n_filters: int = 64,
                 padding: str = 'valid',
                 batch_norm: bool = True,
                 activation: str = 'relu',
                 learning_rate: float = 0.000005,
                 opt_name: str = 'adam',
                 l2_lambda: float = 0.001,
                 loss_weights: List[float] = [1.0, 0.0],
                 merge_type: str = 'concat') -> None:

        """
        Parameters
        ----------
        input_shape : List[int]
            the input shape of the model for both branches

        n_classes: int
            the possible number of classes in the output of the model

        hook_indexes: List[int]
            the respective depths (starting from 0) of hooking [from, to] in the decoders

        depth: int
            the depth of the encoder-decoder branches
        
        n_convs: int
            the number of 2D convolutions per convolutional block

        filter_size: int
            the size of the filter in a 2D convolution

        n_filters: intv
            the number of starting filters (will be increased and decreased by a factor 2 in each conv block in the encoders and decoders, respectively)
            
        padding: str
            padding type in 2D convolution (either 'same' or 'valid')
        
        batch_norm: bool
            boolean for using batch normalization

        activation: str
            activation function after 2D convolution
        
        learning_rate: float
            learning rate of the optimizer

        opt_name: str
            optimizer name (either 'sgd' or 'adam')
        
        l2_lambda: float
            l2 value for regulizer 
        
        loss_weights: bool
            loss contribution for each branch
        
        merge_type: str
            method used for combining feature maps (either 'concat', 'add', 'subtract', 'multiply')
        """

        self._input_shape = input_shape
        self._n_classes = n_classes
        self._hook_indexes = {depth - hook_indexes[0]: hook_indexes[1]}
        self._depth = depth
        self._n_convs = n_convs
        self._filter_size = filter_size
        self._n_filters = n_filters
        self._padding = padding
        self._batch_norm = batch_norm
        self._activation = activation
        self._learning_rate = learning_rate
        self._opt_name = opt_name
        self._l2_lambda = l2_lambda
        self._loss_weights = loss_weights
        self._merge_type = merge_type

        # determine multi-loss model from loss weights
        self._multi_loss = any(loss_weights[1:])

        # set l2 regulizer
        self._l2 = regularizers.l2(self._l2_lambda)

        # placeholder for output_shape
        self._output_shape = []

        # construct model
        self._construct_hooknet_no_list()

    @property
    def input_shape(self) -> List[int]:
        """Return the input shape of the model"""

        return self._input_shape

    @property
    def output_shape(self) -> List[int]:
        """Return the output shape of the model before flattening"""

        return self._output_shape

    def multi_loss(self) -> bool:
        return self._multi_loss

    def _construct_hooknet_default(self) -> None:
        """Construction of single/multi-loss model with multiple inputs and single/multiple outputs"""
        ################################
        # Code for input without lists #
        ################################

        input_1 = Input(self._input_shape)
        input_2 = Input(self._input_shape)

        # construction of context branch and context hooks
        flatten2, context_hooks = self._construct_branch(input_2, reshape_name='reshape_context')

        # construction of target branch with context hooks
        flatten1, _ = self._construct_branch(input_1, context_hooks, reshape_name='reshape_target')

        # create single/multi loss model
        if self._multi_loss:
            # self._create_model([input_1, input_2], [flatten1, flatten2])
            self._create_model([input_1, input_2], [flatten1, flatten2])
        else:
            # self._create_model([input_1, input_2], flatten1)
            self._create_model([input_1, input_2], flatten1)

    def _construct_hooknet_no_list(self) -> None:
        """Construction of single/multi-loss model with multiple inputs and single/multiple outputs"""
        #############################
        # Code for input with lists #
        #############################

        input_total = Input([2] + self._input_shape)

        branch_outputs = []
        for i in range(2):
            # Slicing the ith channel:
            out = Lambda(lambda x: x[:, i])(input_total)
            branch_outputs.append(out)

        input_1, input_2 = branch_outputs

        # construction of context branch and context hooks
        flatten2, context_hooks = self._construct_branch(input_2, reshape_name='reshape_context')

        # construction of target branch with context hooks
        flatten1, _ = self._construct_branch(input_1, context_hooks, reshape_name='reshape_target')

        # flatten1 = Reshape((1, self._output_shape[0] * self._output_shape[1], self._n_classes))(flatten1)
        # flatten2 = Reshape((1, self._output_shape[0] * self._output_shape[1], self._n_classes))(flatten2)
        # flatten_total = Concatenate(axis=1)([flatten1, flatten2])

        # create single/multi loss model
        if self._multi_loss:
            # self._create_model([input_1, input_2], [flatten1, flatten2])
            self._create_model(input_total, flatten_total)
        else:
            # self._create_model([input_1, input_2], flatten1)
            self._create_model(input_total, flatten1)

    def _construct_branch(self,
                          input: Input,
                          in_hooks: Dict = {},
                          reshape_name: str = 'reshape_target') -> Tuple[Tensor, Dict]:
        """
        Construction of single branch
        
        Parameters
        ----------
        input : Input 
            keras Input Tensor
        in_hooks : Dict
            A mapping for hooking from the context branch to the target branch
        reshape_name: str
            name for Reshape Tensor 
        

        Returns
        -------
        flatten: Tensor
            last Tensor of the branch
        out_hooks: Dict
            mapping for hooking between branches
        """

        # input
        net = input

        # encode and retreive residuals
        net, residuals = self._encode_path(net)

        # mid conv block    
        net = self._conv_block(net, self._n_filters * 2 * (self._depth + 1))

        # decode and retreive hooks
        net, out_hooks = self._decode_path(net, residuals, in_hooks)

        # softmax output    
        net = Conv2D(self._n_classes, 1, activation='softmax')(net)

        # set output shape
        self._output_shape = K.int_shape(net)[1:]

        # Reshape net
        flatten = Reshape((self.output_shape[0] * self.output_shape[1], self.output_shape[2]), name=reshape_name)(net)

        # return flatten output and hooks
        return flatten, out_hooks

    def _create_model(self,
                      inputs: Input,
                      outputs: List[Tensor]) -> None:
        """
        Creation of model

        Parameters
        ----------
        inputs: List[Input]
            inputs to the context and target branch
        output: List[Reshape]
            output(s) of the (context) and target branch

        """

        # initilization of keras model
        super().__init__(inputs, outputs)

        # set losses and loss weigths 
        losses = {'reshape_target': 'categorical_crossentropy',
                  'reshape_context': 'categorical_crossentropy'} if self._multi_loss else {
            'reshape_target': 'categorical_crossentropy'}
        loss_weights = {'reshape_target': self._loss_weights[0],
                        'reshape_context': self._loss_weights[1]} if self._multi_loss else {
            'reshape_target': self._loss_weights[0]}

        # compile model
        self.compile(optimizer=self._opt(), loss=losses, loss_weights=loss_weights, metrics=['accuracy'])

        # add output predictions as a metric
        if self._multi_loss:
            self.metrics_tensors += [self.outputs[0]]
        else:
            self.metrics_tensors += self.outputs
        self.metrics_names += ['predictions']

    def _opt(self) -> Optimizer:
        """
        Set optimizer

        Returns
        -------
        SGD or ADAM optimizer

        Raises
        ------

        ValueError: unsupported optimizer

        """

        # Set Gradient-descent optimizer
        if self._opt_name == 'sgd':
            return SGD(lr=self._learning_rate)

        # Set Adam optimizer
        if self._opt_name == 'adam':
            return Adam(lr=self._learning_rate)

        raise ValueError(f'unsupported optimizer name: {self._opt_name}')

    def _encode_path(self, net) -> Tuple[Tensor, List[Tensor]]:
        """
        Encoder
        
        Parameters
        ----------

        net: Tensor
            current Tensor in the model

        
        Returns
        -------
        net: Tensor
            current Tensor in the model
        residuals: List[Tensors]
            all the Tensors used residuals/skip connections in the decoder part of the model

        """

        # list for keeping track for residuals/skip connections
        residuals = []

        # set start filtersize
        n_filters = self._n_filters

        # loop through depths
        for b in range(self._depth):
            # apply convblock
            net = self._conv_block(net, n_filters)

            # keep Tensor for residual/sip connection
            residuals.append(net)

            # downsample
            net = self._downsample(net)

            # increase number of filters with factor 2
            n_filters *= 2

        return net, residuals

    def _decode_path(self,
                     net: Tensor,
                     residuals: List,
                     inhooks: Dict = {}) -> Tensor:
        """
        Decoder

        Parameters
        ----------
        net: Tensor
            current Tensor in the model
        residuals: List[Tensors]
            all the Tensors used residuals/skip connections in the decoder part of the model
        in_hooks: Dict
            mapping for hooking between branches

        Returns
        -------
        net: Tensor
            current Tensor in the model
        hooks: Dict
            mapping between index and Tensor in model for hooking between branches

        """

        # list for keeping potential hook Tensors
        outhooks = []

        # set start number of filters of decoder
        n_filters = self._n_filters * 2 * self._depth

        # loop through depth in reverse
        for b in reversed(range(self._depth)):

            # hook if hook is available
            if b in inhooks:
                # combine feature maps via merge type
                if self._merge_type == 'concat':
                    net = self._concatenator(net, inhooks[b])
                else:
                    net = self._merger(net, inhooks[b])

            # upsample
            net = self._upsample(net, n_filters)

            # concatenate residuals/skip connections
            net = self._concatenator(net, residuals[b])

            # apply conv block
            net = self._conv_block(net, n_filters)

            # set potential hook
            outhooks.append(net)

            n_filters = n_filters // 2

        # get hooks from potential hooks
        hooks = {}
        for shook, ehook in self._hook_indexes.items():
            hooks[ehook] = outhooks[shook]

        print(type(net))
        return net, hooks

    def _conv_block(self, net: Tensor, n_filters: int, kernel_size: int = 3) -> Tensor:
        """
        Convolutional Block

        Parameters
        ----------

        net: Tensor
            current Tensor in the model
        n_filters: int
            current number of filters
        kernel_size: int:
            size of filter in 2d convolution
        
        
        Returns
        -------

        net: Tensor
            current Tensor of the model
        """

        # loop through number of convolutions in convolution block
        for n in range(self._n_convs):
            # apply 2D convolution
            net = Conv2D(n_filters,
                         kernel_size,
                         activation=self._activation,
                         kernel_initializer='he_normal',
                         padding=self._padding,
                         kernel_regularizer=self._l2)(net)

            # apply batch normalization
            if self._batch_norm:
                net = BatchNormalization()(net)

        return net

    def _downsample(self, net: Tensor) -> Tensor:
        """Downsampling via max pooling"""

        return MaxPooling2D(pool_size=(2, 2))(net)

    def _upsample(self, net: Tensor, n_filters: int) -> Tensor:
        """Upsamplign via nearest neightbour interpolation and additional convolution"""

        net = UpSampling2D(size=(2, 2))(net)
        net = Conv2D(n_filters,
                     self._filter_size,
                     activation=self._activation,
                     padding=self._padding,
                     kernel_regularizer=self._l2)(net)

        return net

    def _concatenator(self, net: Tensor, item: Tensor) -> Tensor:
        """"Concatenate feature maps"""

        # crop feature maps
        crop_size = int(item.shape[1] - net.shape[1]) / 2
        item_cropped = Cropping2D(int(crop_size))(item)

        return concatenate([item_cropped, net], axis=3)

    def _merger(self, net: Tensor, item: Tensor) -> Tensor:
        """"Combine feature maps"""

        # crop feature maps
        crop_size = int(item.shape[1] - net.shape[1]) / 2
        item_cropped = Cropping2D(int(crop_size))(item)

        # adapt number of filters via 1x1 convolutional to allow merge
        current_filters = int(net.shape[-1])
        item_cropped = Conv2D(current_filters,
                              1,
                              activation=self._activation,
                              padding=self._padding)(item_cropped)

        # Combine feature maps by adding
        if self._merge_type == 'add':
            return Add()([item_cropped, net])
        # Combine feature maps by subtracting
        if self._merge_type == 'subtract':
            return Subtract()([item_cropped, net])
        # Combine feature maps by multiplication
        if self._merge_type == 'multiply':
            return Multiply()([item_cropped, net])

        # Raise ValueError if merge type is unsupported
        raise ValueError(f'unsupported merge type: {self._merge_type}')
