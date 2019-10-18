"""Inception-v1 Inflated 3D ConvNet used for Kinetics CVPR paper.
The model is introduced in:
  Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
  Joao Carreira, Andrew Zisserman
  https://arxiv.org/pdf/1705.07750v1.pdf.
"""

from __future__ import absolute_import, print_function, division

import warnings

from keras import backend as K
from keras import layers, Model
from keras.utils import get_source_inputs, get_file
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from os.path import exists


def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        weights=None):
    """Internal utility to compute/validate a model's input shape.
    # Arguments
        input_shape: Either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_size: Default input width/height for the model.
        min_size: Minimum input width/height accepted by the model.
        data_format: Image data format to use.
        require_flatten: Whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: One of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
            If weights='imagenet' input channels must be equal to 3.
    # Returns
        An integer shape tuple (may include None entries).
    # Raises
        ValueError: In case of invalid argument values.
    """
    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            default_shape = (input_shape[0], default_size, default_size)
        else:
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting`include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape
    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                   (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                   (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape


"""Inception V3 model for Keras.

Note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function is also different (same as Xception).

# Reference

- [Rethinking the Inception Architecture for Computer Vision](
    http://arxiv.org/abs/1512.00567)

"""


def conv2d_bn_v3(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        kernel_initializer='he_normal',
        use_bias=False,
        name=conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x


def InceptionV3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                disentangled=True,
                **kwargs):
    """Instantiates the Inception v3 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=75,
        data_format=K.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    num_mods = int(input_shape[channel_axis-1]/3)

    if input_tensor is None and disentangled:
        mod_list = list()
        for mod_ind in range(num_mods):
            if channel_axis == 3:
                mod_input = layers.Lambda(lambda x: x[:, :, :, 3*mod_ind:3*mod_ind+3],
                                          output_shape=input_shape[0:2] + [3],
                                          name='mod_input_{}'.format(mod_ind))(img_input)
            else:
                mod_input = layers.Lambda(lambda x: x[:, 3*mod_ind:3*mod_ind+3, :, :],
                                          output_shape=[3] + input_shape[1:3],
                                          name='mod_input_{}'.format(mod_ind))(img_input)
            mod_list.append(conv2d_bn_v3(mod_input, 32, 3, 3, strides=(2, 2), padding='valid',
                                         name='mod_{}'.format(mod_ind)))
        x = layers.Concatenate(name='conv_2d_1')(mod_list)
    else:
        x = conv2d_bn_v3(img_input, 32, 3, 3, strides=(2, 2), padding='valid')

    x = conv2d_bn_v3(x, 32, 3, 3, padding='valid')
    x = conv2d_bn_v3(x, 64, 3, 3)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn_v3(x, 80, 1, 1, padding='valid')
    x = conv2d_bn_v3(x, 192, 3, 3, padding='valid')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn_v3(x, 64, 1, 1)

    branch5x5 = conv2d_bn_v3(x, 48, 1, 1)
    branch5x5 = conv2d_bn_v3(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn_v3(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn_v3(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn_v3(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn_v3(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn_v3(x, 64, 1, 1)

    branch5x5 = conv2d_bn_v3(x, 48, 1, 1)
    branch5x5 = conv2d_bn_v3(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn_v3(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn_v3(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn_v3(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn_v3(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn_v3(x, 64, 1, 1)

    branch5x5 = conv2d_bn_v3(x, 48, 1, 1)
    branch5x5 = conv2d_bn_v3(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn_v3(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn_v3(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn_v3(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn_v3(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn_v3(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn_v3(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn_v3(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn_v3(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn_v3(x, 192, 1, 1)

    branch7x7 = conv2d_bn_v3(x, 128, 1, 1)
    branch7x7 = conv2d_bn_v3(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn_v3(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn_v3(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn_v3(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn_v3(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn_v3(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn_v3(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn_v3(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn_v3(x, 192, 1, 1)

        branch7x7 = conv2d_bn_v3(x, 160, 1, 1)
        branch7x7 = conv2d_bn_v3(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn_v3(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn_v3(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn_v3(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn_v3(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn_v3(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn_v3(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_v3(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn_v3(x, 192, 1, 1)

    branch7x7 = conv2d_bn_v3(x, 192, 1, 1)
    branch7x7 = conv2d_bn_v3(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn_v3(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn_v3(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn_v3(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn_v3(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn_v3(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn_v3(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn_v3(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn_v3(x, 192, 1, 1)
    branch3x3 = conv2d_bn_v3(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn_v3(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn_v3(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn_v3(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn_v3(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn_v3(x, 320, 1, 1)

        branch3x3 = conv2d_bn_v3(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn_v3(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn_v3(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn_v3(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn_v3(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn_v3(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn_v3(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_v3(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='inception_v3')

    return model



"""Inception V3 model for Keras.

Note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function is also different (same as Xception).

# Reference

- [Rethinking the Inception Architecture for Computer Vision](
    http://arxiv.org/abs/1512.00567)

"""



def _obtain_input_shape_3d(input_shape,
                            default_slice_size,
                            min_slice_size,
                            default_num_slices,
                            min_num_slices,
                            data_format,
                            require_flatten,
                            weights=None):
    """Internal utility to compute/validate the model's input shape.
    (Adapted from `keras/applications/imagenet_utils.py`)
    # Arguments
        input_shape: either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_slice_size: default input slices(images) width/height for the model.
        min_slice_size: minimum input slices(images) width/height accepted by the model.
        default_num_slices: default input number of slices(images) for the model.
        min_num_slices: minimum input number of slices accepted by the model.
        data_format: image data format to use.
        require_flatten: whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: one of `None` (random initialization)
            or 'kinetics_only' (pre-training on Kinetics dataset).
            or 'imagenet_and_kinetics' (pre-training on ImageNet and Kinetics datasets).
            If weights='kinetics_only' or weights=='imagenet_and_kinetics' then
            input channels must be equal to 3.
    # Returns
        An integer shape tuple (may include None entries).
    # Raises
        ValueError: in case of invalid argument values.
    """
    if weights != 'kinetics_only' and weights != 'imagenet_and_kinetics' and input_shape and len(input_shape) == 4:
        if data_format == 'channels_first':
            default_shape = (input_shape[2], default_slice_size, default_slice_size, default_num_slices)
        else:
            default_shape = (default_slice_size, default_slice_size, default_num_slices, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_slice_size, default_slice_size, default_num_slices)
        else:
            default_shape = (default_slice_size, default_slice_size, default_num_slices, 3)
    if (weights == 'kinetics_only' or weights == 'imagenet_and_kinetics') and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting`include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape

    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 4:
                    raise ValueError(
                        '`input_shape` must be a tuple of four integers.')

                if input_shape[3] is not None and input_shape[3] < min_num_slices:
                    raise ValueError('Input number of slices must be at least ' +
                                     str(min_num_slices) + '; got '
                                                           '`input_shape=' + str(input_shape) + '`')

                if ((input_shape[1] is not None and input_shape[1] < min_slice_size) or
                        (input_shape[2] is not None and input_shape[2] < min_slice_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_slice_size) + 'x' + str(min_slice_size) + '; got '
                                                                                       '`input_shape=' + str(
                        input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 4:
                    raise ValueError(
                        '`input_shape` must be a tuple of four integers.')

                if input_shape[2] is not None and input_shape[2] < min_num_slices:
                    raise ValueError('Input number of slices must be at least ' +
                                     str(min_num_slices) + '; got '
                                                           '`input_shape=' + str(input_shape) + '`')

                if ((input_shape[0] is not None and input_shape[0] < min_slice_size) or
                        (input_shape[1] is not None and input_shape[1] < min_slice_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_slice_size) + 'x' + str(min_slice_size) + '; got '
                                                                                       '`input_shape=' + str(
                        input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = [3, None, None, None]
            else:
                input_shape = [None, None, None, 3]
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape


def conv3d_bn_v3(x,
                 filters,
                 num_row,
                 num_col,
                 num_dep,
                 padding='same',
                 strides=(1, 1, 1),
                 name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 4
    x = layers.Conv3D(
        filters, (num_row, num_col, num_dep),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x


def Inflated_Inceptionv3(include_top=True,
                         weights='imagenet',
                         input_tensor=None,
                         input_shape=None,
                         pooling=None,
                         classes=1000,
                         disentangled=True,
                         **kwargs):
    """Instantiates the Inception v3 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if not (weights in {'imagenet', None} or exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape_3d(
        input_shape,
        default_slice_size=224,
        min_slice_size=32,
        default_num_slices=5,
        min_num_slices=1,
        data_format=K.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 4

    num_mods = int(input_shape[channel_axis-1]/3)
    if input_shape[2] >= 3 or input_shape[2] is None:
        first_conv_kernel_depth = 3
    else:
        first_conv_kernel_depth = input_shape[2]

    if input_tensor is None and disentangled:
        mod_list = list()
        for mod_ind in range(num_mods):
            if channel_axis == 4:
                mod_input = layers.Lambda(lambda x: x[:, :, :, :, 3*mod_ind:3*mod_ind+3],
                                          output_shape=input_shape[0:3] + [3],
                                          name='mod_input_{}'.format(mod_ind))(img_input)
            else:
                mod_input = layers.Lambda(lambda x: x[:, 3*mod_ind:3*mod_ind+3, :, :, :],
                                          output_shape=[3] + input_shape[1:4],
                                          name='mod_input_{}'.format(mod_ind))(img_input)
            mod_list.append(conv3d_bn_v3(mod_input, 32, 3, 3, first_conv_kernel_depth, strides=(2, 2, 1),
                                         name='mod_{}'.format(mod_ind)))
        x = layers.Concatenate(name='conv_2d_1')(mod_list)
    else:
        x = conv3d_bn_v3(img_input, 32, 3, 3, first_conv_kernel_depth, strides=(2, 2, 1))

    x = conv3d_bn_v3(x, 32, 3, 3, 3)
    x = conv3d_bn_v3(x, 64, 3, 3, 3)
    x = layers.MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding='same')(x)

    x = conv3d_bn_v3(x, 80, 1, 1, 1)
    x = conv3d_bn_v3(x, 192, 3, 3, 3)
    x = layers.MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding='same')(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv3d_bn_v3(x, 64, 1, 1, 1)

    branch5x5 = conv3d_bn_v3(x, 48, 1, 1, 1)
    branch5x5 = conv3d_bn_v3(branch5x5, 64, 5, 5, 5)

    branch3x3dbl = conv3d_bn_v3(x, 64, 1, 1, 1)
    branch3x3dbl = conv3d_bn_v3(branch3x3dbl, 96, 3, 3, 3)
    branch3x3dbl = conv3d_bn_v3(branch3x3dbl, 96, 3, 3, 3)

    branch_pool = layers.AveragePooling3D((3, 3, 3),
                                          strides=(1, 1, 1),
                                          padding='same')(x)
    branch_pool = conv3d_bn_v3(branch_pool, 32, 1, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv3d_bn_v3(x, 64, 1, 1, 1)

    branch5x5 = conv3d_bn_v3(x, 48, 1, 1, 1)
    branch5x5 = conv3d_bn_v3(branch5x5, 64, 5, 5, 5)

    branch3x3dbl = conv3d_bn_v3(x, 64, 1, 1, 1)
    branch3x3dbl = conv3d_bn_v3(branch3x3dbl, 96, 3, 3, 3)
    branch3x3dbl = conv3d_bn_v3(branch3x3dbl, 96, 3, 3, 3)

    branch_pool = layers.AveragePooling3D((3, 3, 3),
                                          strides=(1, 1, 1),
                                          padding='same')(x)
    branch_pool = conv3d_bn_v3(branch_pool, 64, 1, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv3d_bn_v3(x, 64, 1, 1, 1)

    branch5x5 = conv3d_bn_v3(x, 48, 1, 1, 1)
    branch5x5 = conv3d_bn_v3(branch5x5, 64, 5, 5, 5)

    branch3x3dbl = conv3d_bn_v3(x, 64, 1, 1, 1)
    branch3x3dbl = conv3d_bn_v3(branch3x3dbl, 96, 3, 3, 3)
    branch3x3dbl = conv3d_bn_v3(branch3x3dbl, 96, 3, 3, 3)

    branch_pool = layers.AveragePooling3D((3, 3, 3),
                                          strides=(1, 1, 1),
                                          padding='same')(x)
    branch_pool = conv3d_bn_v3(branch_pool, 64, 1, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv3d_bn_v3(x, 384, 3, 3, 3, strides=(2, 2, 1))

    branch3x3dbl = conv3d_bn_v3(x, 64, 1, 1, 1)
    branch3x3dbl = conv3d_bn_v3(branch3x3dbl, 96, 3, 3, 3)
    branch3x3dbl = conv3d_bn_v3(
        branch3x3dbl, 96, 3, 3, 3, strides=(2, 2, 1))

    branch_pool = layers.MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding='same')(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv3d_bn_v3(x, 192, 1, 1, 1)

    branch7x7 = conv3d_bn_v3(x, 128, 1, 1, 1)
    branch7x7 = conv3d_bn_v3(branch7x7, 128, 1, 7, 1)
    branch7x7 = conv3d_bn_v3(branch7x7, 192, 7, 1, 1)

    branch7x7dbl = conv3d_bn_v3(x, 128, 1, 1, 1)
    branch7x7dbl = conv3d_bn_v3(branch7x7dbl, 128, 7, 1, 1)
    branch7x7dbl = conv3d_bn_v3(branch7x7dbl, 128, 1, 7, 1)
    branch7x7dbl = conv3d_bn_v3(branch7x7dbl, 128, 7, 1, 1)
    branch7x7dbl = conv3d_bn_v3(branch7x7dbl, 192, 1, 7, 1)

    branch_pool = layers.AveragePooling3D((3, 3, 3),
                                          strides=(1, 1, 1),
                                          padding='same')(x)
    branch_pool = conv3d_bn_v3(branch_pool, 192, 1, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv3d_bn_v3(x, 192, 1, 1, 1)

        branch7x7 = conv3d_bn_v3(x, 160, 1, 1, 1)
        branch7x7 = conv3d_bn_v3(branch7x7, 160, 1, 7, 1)
        branch7x7 = conv3d_bn_v3(branch7x7, 192, 7, 1, 1)

        branch7x7dbl = conv3d_bn_v3(x, 160, 1, 1, 1)
        branch7x7dbl = conv3d_bn_v3(branch7x7dbl, 160, 7, 1, 1)
        branch7x7dbl = conv3d_bn_v3(branch7x7dbl, 160, 1, 7, 1)
        branch7x7dbl = conv3d_bn_v3(branch7x7dbl, 160, 7, 1, 1)
        branch7x7dbl = conv3d_bn_v3(branch7x7dbl, 192, 1, 7, 1)

        branch_pool = layers.AveragePooling3D(
            (3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        branch_pool = conv3d_bn_v3(branch_pool, 192, 1, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv3d_bn_v3(x, 192, 1, 1, 1)

    branch7x7 = conv3d_bn_v3(x, 192, 1, 1, 1)
    branch7x7 = conv3d_bn_v3(branch7x7, 192, 1, 7, 1)
    branch7x7 = conv3d_bn_v3(branch7x7, 192, 7, 1, 1)

    branch7x7dbl = conv3d_bn_v3(x, 192, 1, 1, 1)
    branch7x7dbl = conv3d_bn_v3(branch7x7dbl, 192, 7, 1, 1)
    branch7x7dbl = conv3d_bn_v3(branch7x7dbl, 192, 1, 7, 1)
    branch7x7dbl = conv3d_bn_v3(branch7x7dbl, 192, 7, 1, 1)
    branch7x7dbl = conv3d_bn_v3(branch7x7dbl, 192, 1, 7, 1)

    branch_pool = layers.AveragePooling3D((3, 3, 3),
                                          strides=(1, 1, 1),
                                          padding='same')(x)
    branch_pool = conv3d_bn_v3(branch_pool, 192, 1, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv3d_bn_v3(x, 192, 1, 1, 1)
    branch3x3 = conv3d_bn_v3(branch3x3, 320, 3, 3, 3,
                             strides=(2, 2, 1))

    branch7x7x3 = conv3d_bn_v3(x, 192, 1, 1, 1)
    branch7x7x3 = conv3d_bn_v3(branch7x7x3, 192, 1, 7, 1)
    branch7x7x3 = conv3d_bn_v3(branch7x7x3, 192, 7, 1, 1)
    branch7x7x3 = conv3d_bn_v3(
        branch7x7x3, 192, 3, 3, 3, strides=(2, 2, 1))

    branch_pool = layers.MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding='same')(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv3d_bn_v3(x, 320, 1, 1, 1)

        branch3x3 = conv3d_bn_v3(x, 384, 1, 1, 1)
        branch3x3_1 = conv3d_bn_v3(branch3x3, 384, 1, 3, 1)
        branch3x3_2 = conv3d_bn_v3(branch3x3, 384, 3, 1, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv3d_bn_v3(x, 448, 1, 1, 1)
        branch3x3dbl = conv3d_bn_v3(branch3x3dbl, 384, 3, 3, 3)
        branch3x3dbl_1 = conv3d_bn_v3(branch3x3dbl, 384, 1, 3, 1)
        branch3x3dbl_2 = conv3d_bn_v3(branch3x3dbl, 384, 3, 1, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling3D(
            (3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        branch_pool = conv3d_bn_v3(branch_pool, 192, 1, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling3D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling3D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling3D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='inflated_inception_v3')

    return model



"""Inception-ResNet V2 model for Keras.

Model naming and structure follows TF-slim implementation
(which has some additional layers and different number of
filters from the original arXiv paper):
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py

Pre-trained ImageNet weights are also converted from TF-slim,
which can be found in:
https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models

# Reference
- [Inception-v4, Inception-ResNet and the Impact of
   Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

"""

BASE_WEIGHT_URL = ('https://github.com/fchollet/deep-learning-models/'
                   'releases/download/v0.7/')


def conv2d_bn_v4(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = layers.Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = layers.BatchNormalization(axis=bn_axis,
                                      scale=False,
                                      name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = layers.Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds a Inception-ResNet block.

    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`

    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch.
            Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names.
            The Inception-ResNet blocks
            are repeated many times in this network.
            We use `block_idx` to identify
            each of the repetitions. For example,
            the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`,
            and the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](../activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).

    # Returns
        Output tensor for the block.

    # Raises
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    """
    if block_type == 'block35':
        branch_0 = conv2d_bn_v4(x, 32, 1)
        branch_1 = conv2d_bn_v4(x, 32, 1)
        branch_1 = conv2d_bn_v4(branch_1, 32, 3)
        branch_2 = conv2d_bn_v4(x, 32, 1)
        branch_2 = conv2d_bn_v4(branch_2, 48, 3)
        branch_2 = conv2d_bn_v4(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn_v4(x, 192, 1)
        branch_1 = conv2d_bn_v4(x, 128, 1)
        branch_1 = conv2d_bn_v4(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn_v4(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn_v4(x, 192, 1)
        branch_1 = conv2d_bn_v4(x, 192, 1)
        branch_1 = conv2d_bn_v4(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn_v4(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    mixed = layers.Concatenate(
        axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_bn_v4(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv')

    x = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=K.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=block_name)([x, up])
    if activation is not None:
        x = layers.Activation(activation, name=block_name + '_ac')(x)
    return x


def InceptionResNetV2(include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=1000,
                      **kwargs):
    """Instantiates the Inception-ResNet v2 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(299, 299, 3)` (with `'channels_last'` data format)
            or `(3, 299, 299)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional block.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.

    # Returns
        A Keras `Model` instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=75,
        data_format=K.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Stem block: 35 x 35 x 192
    x = conv2d_bn_v4(img_input, 32, 3, strides=2, padding='valid')
    x = conv2d_bn_v4(x, 32, 3, padding='valid')
    x = conv2d_bn_v4(x, 64, 3)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn_v4(x, 80, 1, padding='valid')
    x = conv2d_bn_v4(x, 192, 3, padding='valid')
    x = layers.MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn_v4(x, 96, 1)
    branch_1 = conv2d_bn_v4(x, 48, 1)
    branch_1 = conv2d_bn_v4(branch_1, 64, 5)
    branch_2 = conv2d_bn_v4(x, 64, 1)
    branch_2 = conv2d_bn_v4(branch_2, 96, 3)
    branch_2 = conv2d_bn_v4(branch_2, 96, 3)
    branch_pool = layers.AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn_v4(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    x = layers.Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn_v4(x, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn_v4(x, 256, 1)
    branch_1 = conv2d_bn_v4(branch_1, 256, 3)
    branch_1 = conv2d_bn_v4(branch_1, 384, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn_v4(x, 256, 1)
    branch_0 = conv2d_bn_v4(branch_0, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn_v4(x, 256, 1)
    branch_1 = conv2d_bn_v4(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = conv2d_bn_v4(x, 256, 1)
    branch_2 = conv2d_bn_v4(branch_2, 288, 3)
    branch_2 = conv2d_bn_v4(branch_2, 320, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn_v4(x, 1536, 1, name='conv_7b')

    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='inception_resnet_v2')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            fname = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
            weights_path = get_file(
                fname,
                BASE_WEIGHT_URL + fname,
                cache_subdir='models',
                file_hash='e693bd0210a403b3192acc6073ad2e96')
        else:
            fname = ('inception_resnet_v2_weights_'
                     'tf_dim_ordering_tf_kernels_notop.h5')
            weights_path = get_file(
                fname,
                BASE_WEIGHT_URL + fname,
                cache_subdir='models',
                file_hash='d19885ff4a710c122648d3b5c3b684e4')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model




"""Inception-v1 Inflated 3D ConvNet used for Kinetics CVPR paper.

The model is introduced in:

Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
Joao Carreira, Andrew Zisserman
https://arxiv.org/abs/1705.07750v1
"""

WEIGHTS_NAME = ['rgb_kinetics_only', 'flow_kinetics_only', 'rgb_imagenet_and_kinetics', 'flow_imagenet_and_kinetics']

# path to pretrained models with top (classification layer)
WEIGHTS_PATH_I3D = {
    'rgb_kinetics_only': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'flow_kinetics_only': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'rgb_imagenet_and_kinetics': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5',
    'flow_imagenet_and_kinetics': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
}

# path to pretrained models with no top (no classification layer)
WEIGHTS_PATH_NO_TOP_I3D = {
    'rgb_kinetics_only': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_kinetics_only': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'rgb_imagenet_and_kinetics': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_imagenet_and_kinetics': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
}


def conv3d_bn(x,
              filters,
              num_row,
              num_col,
              num_slices,
              padding='same',
              strides=(1, 1, 1),
              use_bias=False,
              use_activation_fn=True,
              use_bn=True,
              name=None):
    """Utility function to apply conv3d + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv3D`.
        num_slices: slices (time depth) of the convolution kernel.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv3D`.
        strides: strides in `Conv3D`.
        use_bias: use bias or not
        use_activation_fn: use an activation function or not.
        use_bn: use batch normalization or not.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv3D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = layers.Conv3D(
        filters, (num_row, num_col, num_slices),
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=conv_name)(x)

    if use_bn:
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 4
        x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    if use_activation_fn:
        x = layers.Activation('relu', name=name)(x)

    return x


def Inception_Inflated3d(include_top=True,
                         weights=None,
                         input_tensor=None,
                         input_shape=None,
                         dropout_prob=0.0,
                         endpoint_logit=True,
                         disentangled = True,
                         classes=400):
    """Instantiates the Inflated 3D Inception v1 architecture.
    Optionally loads weights pre-trained
    on Kinetics. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input slice(image) size for this model is 224x224.
    # Arguments
        include_top: whether to include the the classification
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or 'kinetics_only' (pre-training on Kinetics dataset only).
            or 'imagenet_and_kinetics' (pre-training on ImageNet and Kinetics datasets).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(NUM_FRAMES, 224, 224, 3)` (with `channels_last` data format)
            or `(NUM_FRAMES, 3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
            NUM_FRAMES should be no smaller than 8. The authors used 64
            slices per example for training and testing on kinetics dataset
            Also, Width and height should be no smaller than 32.
            E.g. `(64, 150, 150, 3)` would be one valid value.
        dropout_prob: optional, dropout probability applied in dropout layer
            after global average pooling layer.
            0.0 means no dropout is applied, 1.0 means dropout is applied to all features.
            Note: Since Dropout is applied just before the classification
            layer, it is only useful when `include_top` is set to True.
        endpoint_logit: (boolean) optional. If True, the model's forward pass
            will end at producing logits. Otherwise, softmax is applied after producing
            the logits to produce the class probabilities prediction. Setting this parameter
            to True is particularly useful when you want to combine results of rgb model
            and optical flow model.
            - `True` end model forward pass at logit output
            - `False` go further after logit to produce softmax predictions
            Note: This parameter is only useful when `include_top` is set to True.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in WEIGHTS_NAME or weights is None or exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or %s' %
                         str(WEIGHTS_NAME) + ' '
                                             'or a valid path to a file containing `weights` values')

    if weights in WEIGHTS_NAME and include_top and classes != 400:
        raise ValueError('If using `weights` as one of these %s, with `include_top`'
                         ' as true, `classes` should be 400' % str(WEIGHTS_NAME))

    # Determine proper input shape
    input_shape = _obtain_input_shape_3d(
        input_shape,
        default_slice_size=224,
        min_slice_size=32,
        default_num_slices=5,
        min_num_slices=1,
        data_format=K.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 4

    num_mods = int(input_shape[channel_axis - 1] / 3)
    if input_shape[2] >= 7 or input_shape[2] is None:
        first_conv_kernel_depth = 7
    else:
        first_conv_kernel_depth = input_shape[2]

    if input_tensor is None and disentangled:
        mod_list = list()
        for mod_ind in range(num_mods):
            if channel_axis == 4:
                mod_input = layers.Lambda(lambda x: x[:, :, :, :, 3 * mod_ind:3 * mod_ind + 3],
                                          output_shape=input_shape[0:3] + [3],
                                          name='mod_input_{}'.format(mod_ind))(img_input)
            else:
                mod_input = layers.Lambda(lambda x: x[:, 3 * mod_ind:3 * mod_ind + 3, :, :, :],
                                          output_shape=[3] + input_shape[1:4],
                                          name='mod_input_{}'.format(mod_ind))(img_input)
            mod_list.append(conv3d_bn(mod_input, 64, 7, 7, first_conv_kernel_depth, strides=(2, 2, 1), padding='same',
                                      name='mod_{}'.format(mod_ind)))
        x = layers.concatenate(mod_list, axis=channel_axis, name='Conv3d_1a_7x7')
    else:
        x = conv3d_bn(img_input, 64, 7, 7, first_conv_kernel_depth, strides=(2, 2, 1), padding='same',
                      name='Conv3d_1a_7x7')

    # Downsampling (spatial only)
    x = layers.MaxPooling3D((3, 3, 1), strides=(2, 2, 1), padding='same', name='MaxPool2d_2a_3x3')(x)
    x = conv3d_bn(x, 64, 1, 1, 1, strides=(1, 1, 1), padding='same', name='Conv3d_2b_1x1')
    x = conv3d_bn(x, 192, 3, 3, 3, strides=(1, 1, 1), padding='same', name='Conv3d_2c_3x3')

    # Downsampling (spatial only)
    x = layers.MaxPooling3D((3, 3, 1), strides=(2, 2, 1), padding='same', name='MaxPool2d_3a_3x3')(x)

    # Mixed 3b
    branch_0 = conv3d_bn(x, 64, 1, 1, 1, padding='same', name='Conv3d_3b_0a_1x1')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same', name='Conv3d_3b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 128, 3, 3, 3, padding='same', name='Conv3d_3b_1b_3x3')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same', name='Conv3d_3b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 32, 3, 3, 3, padding='same', name='Conv3d_3b_2b_3x3')

    branch_3 = layers.MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 32, 1, 1, 1, padding='same', name='Conv3d_3b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3b')

    # Mixed 3c
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_0a_1x1')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 192, 3, 3, 3, padding='same', name='Conv3d_3c_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_3c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 96, 3, 3, 3, padding='same', name='Conv3d_3c_2b_3x3')

    branch_3 = layers.MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_3c_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3c')

    # Downsampling (spatial only)
    x = layers.MaxPooling3D((3, 3, 1), strides=(2, 2, 1), padding='same', name='MaxPool2d_4a_3x3')(x)

    # Mixed 4b
    branch_0 = conv3d_bn(x, 192, 1, 1, 1, padding='same', name='Conv3d_4b_0a_1x1')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same', name='Conv3d_4b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 208, 3, 3, 3, padding='same', name='Conv3d_4b_1b_3x3')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same', name='Conv3d_4b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 48, 3, 3, 3, padding='same', name='Conv3d_4b_2b_3x3')

    branch_3 = layers.MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4b')

    # Mixed 4c
    branch_0 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_4c_0a_1x1')

    branch_1 = conv3d_bn(x, 112, 1, 1, 1, padding='same', name='Conv3d_4c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 224, 3, 3, 3, padding='same', name='Conv3d_4c_1b_3x3')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same', name='Conv3d_4c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4c_2b_3x3')

    branch_3 = layers.MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4c_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4c')

    # Mixed 4d
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_0a_1x1')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 256, 3, 3, 3, padding='same', name='Conv3d_4d_1b_3x3')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same', name='Conv3d_4d_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4d_2b_3x3')

    branch_3 = layers.MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4d_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4d_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4d')

    # Mixed 4e
    branch_0 = conv3d_bn(x, 112, 1, 1, 1, padding='same', name='Conv3d_4e_0a_1x1')

    branch_1 = conv3d_bn(x, 144, 1, 1, 1, padding='same', name='Conv3d_4e_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 288, 3, 3, 3, padding='same', name='Conv3d_4e_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_4e_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4e_2b_3x3')

    branch_3 = layers.MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4e_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4e_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4e')

    # Mixed 4f
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same', name='Conv3d_4f_0a_1x1')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_4f_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_4f_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_4f_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_4f_2b_3x3')

    branch_3 = layers.MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4f_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_4f_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4f')

    # Downsampling (spatial only)
    x = layers.MaxPooling3D((2, 2, 1), strides=(2, 2, 1), padding='same', name='MaxPool2d_5a_2x2')(x)

    # Mixed 5b
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same', name='Conv3d_5b_0a_1x1')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_5b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_5b_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_5b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5b_2b_3x3')

    branch_3 = layers.MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_5b')

    # Mixed 5c
    branch_0 = conv3d_bn(x, 384, 1, 1, 1, padding='same', name='Conv3d_5c_0a_1x1')

    branch_1 = conv3d_bn(x, 192, 1, 1, 1, padding='same', name='Conv3d_5c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 384, 3, 3, 3, padding='same', name='Conv3d_5c_1b_3x3')

    branch_2 = conv3d_bn(x, 48, 1, 1, 1, padding='same', name='Conv3d_5c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5c_2b_3x3')

    branch_3 = layers.MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5c_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_5c')

    if include_top:
        # Classification block
        x = layers.AveragePooling3D((7, 7, 2), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)
        x = layers.Dropout(dropout_prob)(x)

        x = conv3d_bn(x, classes, 1, 1, 1, padding='same',
                      use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

        num_slices_remaining = int(x.shape[1])
        x = layers.Reshape((num_slices_remaining, classes))(x)

        # logits (raw scores for each class)
        x = layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                   output_shape=lambda s: (s[0], s[2]))(x)

        if not endpoint_logit:
            x = layers.Activation('softmax', name='prediction')(x)
    else:
        x = layers.GlobalAveragePooling3D(name='global_avg_pool')(x)

    inputs = img_input
    # create model
    model = Model(inputs, x, name='i3d_inception')

    # # load weights
    # if weights in WEIGHTS_NAME:
    #     if weights == WEIGHTS_NAME[0]:  # rgb_kinetics_only
    #         if include_top:
    #             weights_url = WEIGHTS_PATH_I3D['rgb_kinetics_only']
    #             model_name = 'i3d_inception_rgb_kinetics_only.h5'
    #         else:
    #             weights_url = WEIGHTS_PATH_NO_TOP_I3D['rgb_kinetics_only']
    #             model_name = 'i3d_inception_rgb_kinetics_only_no_top.h5'
    #
    #     elif weights == WEIGHTS_NAME[1]:  # flow_kinetics_only
    #         if include_top:
    #             weights_url = WEIGHTS_PATH_I3D['flow_kinetics_only']
    #             model_name = 'i3d_inception_flow_kinetics_only.h5'
    #         else:
    #             weights_url = WEIGHTS_PATH_NO_TOP_I3D['flow_kinetics_only']
    #             model_name = 'i3d_inception_flow_kinetics_only_no_top.h5'
    #
    #     elif weights == WEIGHTS_NAME[2]:  # rgb_imagenet_and_kinetics
    #         if include_top:
    #             weights_url = WEIGHTS_PATH_I3D['rgb_imagenet_and_kinetics']
    #             model_name = 'i3d_inception_rgb_imagenet_and_kinetics.h5'
    #         else:
    #             weights_url = WEIGHTS_PATH_NO_TOP_I3D['rgb_imagenet_and_kinetics']
    #             model_name = 'i3d_inception_rgb_imagenet_and_kinetics_no_top.h5'
    #
    #     elif weights == WEIGHTS_NAME[3]:  # flow_imagenet_and_kinetics
    #         if include_top:
    #             weights_url = WEIGHTS_PATH_I3D['flow_imagenet_and_kinetics']
    #             model_name = 'i3d_inception_flow_imagenet_and_kinetics.h5'
    #         else:
    #             weights_url = WEIGHTS_PATH_NO_TOP_I3D['flow_imagenet_and_kinetics']
    #             model_name = 'i3d_inception_flow_imagenet_and_kinetics_no_top.h5'
    #
    #     downloaded_weights_path = get_file(model_name, weights_url, cache_subdir='models')
    #     model.load_weights(downloaded_weights_path)
    #
    # elif weights is not None:
    #     model.load_weights(weights)


    return model