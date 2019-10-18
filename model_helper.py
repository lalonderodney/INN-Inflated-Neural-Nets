'''
INN: Inflated Neural Networks for IPMN Diagnosis
Original Paper by Rodney LaLonde, Irene Tanner, Katerina Nikiforaki, Georgios Z. Papadakis, Pujan Kandel,
Candice W. Bolan, Michael B. Wallace, Ulas Bagci
(https://link.springer.com/chapter/10.1007/978-3-030-32254-0_12, https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This is a helper file for choosing which model to create.
This file also contains the transfer weights function which is key to making the INN work.
'''
from __future__ import division

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from keras import backend as K
from keras.utils import get_file


def create_model(args, input_shape):
    # Set the activation function for either binary or multi-class
    if args.num_classes == 2:
        n_class = 1
        activ='sigmoid'
    else:
        n_class = args.num_classes
        activ='softmax'

    # If using CPU or single GPU the following code runs
    # Multi-GPU is an exact copy below, but with the model placed on the CPU
    if args.gpus <= 1:
        # Double check the network is one of the ones we're prepared for.
        if args.net.find('inceptionv3') != -1 or args.net.find('densenet') != -1:
            # Find the number of modalities
            if K.image_data_format() == 'channels_first':
                num_mods = int(input_shape[0] / 3)
            else:
                num_mods = int(input_shape[-1] / 3)

            if args.net.find('inceptionv3') != -1:
                if args.net == 'inceptionv3':
                    # Load the 2D Inceptionv3 network
                    from inceptionnets import InceptionV3
                    base_model = InceptionV3(include_top=False, weights=None, input_shape=input_shape,
                                             pooling='avg', classes=args.num_classes, disentangled=args.disentangled)
                    num_dims = 2
                elif args.net == 'inflated_inceptionv3':
                    # Load the InceptINN (3D Inflated Inceptionv3) network
                    from inceptionnets import Inflated_Inceptionv3
                    base_model = Inflated_Inceptionv3(input_shape=input_shape, include_top=False, weights=None,
                                                      pooling='avg', classes=args.num_classes,
                                                      disentangled=args.disentangled)
                    num_dims = 3

                # If we're using pre-trained weights
                if args.use_default_pretrained:
                    from inceptionnets import InceptionV3
                    WEIGHTS_PATH_NO_TOP = (
                        'https://github.com/fchollet/deep-learning-models/'
                        'releases/download/v0.5/'
                        'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
                    weights_path = get_file(
                        'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        WEIGHTS_PATH_NO_TOP,
                        cache_subdir='models',
                        file_hash='bcbd6486424b2319ff4ef7d526e38f63')
                    temp_model = InceptionV3(include_top=False, weights=None, disentangled=False, pooling='avg')
                    last_ind = 4 * num_mods + 2 # This is an offset to handle the multi-modalities

            else:
                if args.net.find('densenet121') != -1:
                    # All the settings for DenseNet121
                    depth = 121
                    growth_rate = 32
                    nb_filter = 64
                    nb_layers_per_block = [6, 12, 24, 16]
                else:
                    raise Exception('Cannot find matching version of densenet')

                if args.net.find('inflated') != -1:
                    # Load the DenseINN (3D Inflated DenseNet121) network
                    from densenets import Inflated_DenseNet
                    base_model = Inflated_DenseNet(input_shape=input_shape, depth=depth, nb_dense_block=4,
                                                   growth_rate=growth_rate, nb_filter=nb_filter,
                                                   nb_layers_per_block=nb_layers_per_block, bottleneck=True,
                                                   reduction=args.reduction, dropout_rate=args.drop_rate,
                                                   weight_decay=args.weight_decay, subsample_initial_block=True,
                                                   include_top=False, weights=None, input_tensor=None, pooling='avg',
                                                   classes=n_class, activation=activ, disentangled=args.disentangled)
                    num_dims = 3

                else:
                    # Load the standard 2D DenseNet121
                    from densenets import DenseNet
                    base_model = DenseNet(input_shape=input_shape, depth=depth, nb_dense_block=4,
                                          growth_rate=growth_rate, nb_filter=nb_filter,
                                          nb_layers_per_block=nb_layers_per_block, bottleneck=True,
                                          reduction=args.reduction, dropout_rate=args.drop_rate,
                                          weight_decay=args.weight_decay, subsample_initial_block=True,
                                          include_top=False, weights=None, input_tensor=None, pooling='avg',
                                          classes=n_class, activation=activ, disentangled=args.disentangled)
                    num_dims = 2

                # If we're using pre-trained weights
                if args.use_default_pretrained:
                    from densenets import DenseNet
                    temp_model = DenseNet(input_shape=input_shape[:2] + [3,], depth=depth, nb_dense_block=4,
                                          growth_rate=growth_rate, nb_filter=nb_filter,
                                          nb_layers_per_block=nb_layers_per_block, bottleneck=True,
                                          reduction=args.reduction, dropout_rate=args.drop_rate,
                                          weight_decay=args.weight_decay, subsample_initial_block=True,
                                          include_top=False, weights=None, input_tensor=None, pooling='avg',
                                          classes=n_class, activation=activ, disentangled=False)
                    DENSENET_121_WEIGHTS_PATH_NO_TOP = \
                        r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-121-32-no-top.h5'
                    weights_path = get_file('DenseNet-BC-121-32-no-top.h5',
                                            DENSENET_121_WEIGHTS_PATH_NO_TOP,
                                            cache_subdir='models',
                                            md5_hash='55e62a6358af8a0af0eedf399b5aea99')
                    last_ind = 4 * num_mods + 3 # This is an offset to handle the multi-modalities

            # Add the prediction layer after the base model
            x = base_model.output
            predictions = Dense(n_class, activation=activ, name='out')(x)

            # Form the model object based on the inputs and outputs
            model = Model(inputs=base_model.input, outputs=predictions)

            # Load the weights based on the weights path above set for each network
            if args.use_default_pretrained:
                temp_model.load_weights(weights_path)
                base_model = transfer_weights(old_model=temp_model, new_model=base_model, two_or_three_dim=num_dims,
                                              disentangled=args.disentangled, num_mods=num_mods, last_ind=last_ind)

            # If we want to load from a custom weights path (for example for resuming training)
            elif args.custom_weights_path != '':
                model.load_weights(args.custom_weights_path)

            # If we want to freeze the base network weights and only train the prediction layer
            if args.freeze_base_weights:
                for layer in base_model.layers:
                    layer.trainable = False

            return model

        else:
            raise Exception('Unknown network type specified: {}'.format(args.net))
    # If using multiple GPUs, this is the exact same as above, but placed onto the CPU
    else:
        with tf.device("/cpu:0"):
            # Double check the network is one of the ones we're prepared for.
            if args.net.find('inceptionv3') != -1 or args.net.find('densenet') != -1:
                # Find the number of modalities
                if K.image_data_format() == 'channels_first':
                    num_mods = int(input_shape[0] / 3)
                else:
                    num_mods = int(input_shape[-1] / 3)

                if args.net.find('inceptionv3') != -1:
                    if args.net == 'inceptionv3':
                        # Load the 2D Inceptionv3 network
                        from inceptionnets import InceptionV3
                        base_model = InceptionV3(include_top=False, weights=None, input_shape=input_shape,
                                                 pooling='avg', classes=args.num_classes,
                                                 disentangled=args.disentangled)
                        num_dims = 2
                    elif args.net == 'inflated_inceptionv3':
                        # Load the InceptINN (3D Inflated Inceptionv3) network
                        from inceptionnets import Inflated_Inceptionv3
                        base_model = Inflated_Inceptionv3(input_shape=input_shape, include_top=False, weights=None,
                                                          pooling='avg', classes=args.num_classes,
                                                          disentangled=args.disentangled)
                        num_dims = 3

                    # If we're using pre-trained weights
                    if args.use_default_pretrained:
                        from inceptionnets import InceptionV3
                        WEIGHTS_PATH_NO_TOP = (
                            'https://github.com/fchollet/deep-learning-models/'
                            'releases/download/v0.5/'
                            'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
                        weights_path = get_file(
                            'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models',
                            file_hash='bcbd6486424b2319ff4ef7d526e38f63')
                        temp_model = InceptionV3(include_top=False, weights=None, disentangled=False, pooling='avg')
                        last_ind = 4 * num_mods + 2  # This is an offset to handle the multi-modalities

                else:
                    if args.net.find('densenet121') != -1:
                        # All the settings for DenseNet121
                        depth = 121
                        growth_rate = 32
                        nb_filter = 64
                        nb_layers_per_block = [6, 12, 24, 16]
                    else:
                        raise Exception('Cannot find matching version of densenet')

                    if args.net.find('inflated') != -1:
                        # Load the DenseINN (3D Inflated DenseNet121) network
                        from densenets import Inflated_DenseNet
                        base_model = Inflated_DenseNet(input_shape=input_shape, depth=depth, nb_dense_block=4,
                                                       growth_rate=growth_rate, nb_filter=nb_filter,
                                                       nb_layers_per_block=nb_layers_per_block, bottleneck=True,
                                                       reduction=args.reduction, dropout_rate=args.drop_rate,
                                                       weight_decay=args.weight_decay, subsample_initial_block=True,
                                                       include_top=False, weights=None, input_tensor=None,
                                                       pooling='avg',
                                                       classes=n_class, activation=activ,
                                                       disentangled=args.disentangled)
                        num_dims = 3

                    else:
                        # Load the standard 2D DenseNet121
                        from densenets import DenseNet
                        base_model = DenseNet(input_shape=input_shape, depth=depth, nb_dense_block=4,
                                              growth_rate=growth_rate, nb_filter=nb_filter,
                                              nb_layers_per_block=nb_layers_per_block, bottleneck=True,
                                              reduction=args.reduction, dropout_rate=args.drop_rate,
                                              weight_decay=args.weight_decay, subsample_initial_block=True,
                                              include_top=False, weights=None, input_tensor=None, pooling='avg',
                                              classes=n_class, activation=activ, disentangled=args.disentangled)
                        num_dims = 2

                    # If we're using pre-trained weights
                    if args.use_default_pretrained:
                        from densenets import DenseNet
                        temp_model = DenseNet(input_shape=input_shape[:2] + [3, ], depth=depth, nb_dense_block=4,
                                              growth_rate=growth_rate, nb_filter=nb_filter,
                                              nb_layers_per_block=nb_layers_per_block, bottleneck=True,
                                              reduction=args.reduction, dropout_rate=args.drop_rate,
                                              weight_decay=args.weight_decay, subsample_initial_block=True,
                                              include_top=False, weights=None, input_tensor=None, pooling='avg',
                                              classes=n_class, activation=activ, disentangled=False)
                        DENSENET_121_WEIGHTS_PATH_NO_TOP = \
                            r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-121-32-no-top.h5'
                        weights_path = get_file('DenseNet-BC-121-32-no-top.h5',
                                                DENSENET_121_WEIGHTS_PATH_NO_TOP,
                                                cache_subdir='models',
                                                md5_hash='55e62a6358af8a0af0eedf399b5aea99')
                        last_ind = 4 * num_mods + 3  # This is an offset to handle the multi-modalities

                # Add the prediction layer after the base model
                x = base_model.output
                predictions = Dense(n_class, activation=activ, name='out')(x)

                # Form the model object based on the inputs and outputs
                model = Model(inputs=base_model.input, outputs=predictions)

                # Load the weights based on the weights path above set for each network
                if args.use_default_pretrained:
                    temp_model.load_weights(weights_path)
                    base_model = transfer_weights(old_model=temp_model, new_model=base_model, two_or_three_dim=num_dims,
                                                  disentangled=args.disentangled, num_mods=num_mods, last_ind=last_ind)

                # If we want to load from a custom weights path (for example for resuming training)
                elif args.custom_weights_path != '':
                    model.load_weights(args.custom_weights_path)

                # If we want to freeze the base network weights and only train the prediction layer
                if args.freeze_base_weights:
                    for layer in base_model.layers:
                        layer.trainable = False

                return model

            else:
                raise Exception('Unknown network type specified: {}'.format(args.net))

def transfer_weights(old_model, new_model, two_or_three_dim=2, disentangled=True, num_mods=2, last_ind=10):
    '''
    This is the key function for INN to work. Here we take the old (usually) 2D model and transfer the weights to
    the new 3D model. Note, this is a little bit messy, as it is more a proof of concept than anything resembling
    production code.
    :param old_model: Model to transfer weights from
    :param new_model: Model to transfer weights to
    :param two_or_three_dim: Number of dimensions in the old model (2D or 3D)
    :param disentangled: The type of fusion strategy we're using (early or intermediate)
    :param num_mods: The number of modalities we have
    :param last_ind: This is an offset to help us keep track of things, set based on model and number of modalities
    :return: new model with transferred weights
    '''
    assert(two_or_three_dim == 2 or two_or_three_dim == 3), 'Choice must be one of 2D or 3D for transferring weights.'

    if disentangled:
        assert len(new_model.layers) == len(
            old_model.layers) + 4 * num_mods - 2, 'Found wrong number of layers for transferring weights'
    else:
        assert len(new_model.layers) == len(
            old_model.layers), 'Found wrong number of layers for transferring weights'

    for layer_ind, layer_val in enumerate(tqdm(new_model.layers, desc='Transferring ImageNet Weights')):
        if not disentangled and layer_ind == 1:
            if two_or_three_dim == 3:
                num_depth = new_model.layers[1].kernel_size[2]
                new_model.layers[1].set_weights(
                    [np.tile(np.expand_dims(old_model.layers[1].get_weights()[0], axis=2) / (2.*num_depth),
                             (1, 1, num_depth, 2, 1))])
            else:
                new_model.layers[1].set_weights([np.tile(old_model.layers[1].get_weights()[0] / 2., (1, 1, 2, 1))])
        elif not disentangled and layer_ind != 1:
            if two_or_three_dim == 3:
                try:
                    num_depth = new_model.layers[layer_ind].kernel_size[2]
                    new_model.layers[layer_ind].set_weights(
                        [np.tile(np.expand_dims(old_model.layers[layer_ind].get_weights()[0], axis=2) / num_depth,
                                 (1, 1, num_depth, 1, 1))])
                except:
                    new_model.layers[layer_ind].set_weights(old_model.layers[layer_ind].get_weights())
            else:
                new_model.layers[layer_ind].set_weights(old_model.layers[layer_ind].get_weights())
        elif disentangled and layer_ind < num_mods + 1:
            continue
        elif disentangled and layer_ind == num_mods + 1:
            for mod_ind in range(num_mods):
                if two_or_three_dim == 3:
                    num_depth = new_model.layers[layer_ind + mod_ind].kernel_size[2]
                    new_model.layers[layer_ind + mod_ind].set_weights(
                        [np.tile(np.expand_dims(old_model.layers[1].get_weights()[0], axis=2) / num_depth, (1, 1, num_depth, 1, 1))])
                else:
                    new_model.layers[layer_ind + mod_ind].set_weights(old_model.layers[1].get_weights())
                new_model.layers[layer_ind + num_mods + mod_ind].set_weights(old_model.layers[2].get_weights())
        elif disentangled and layer_ind < last_ind:
            continue
        else:
            try:
                if two_or_three_dim == 3:
                    num_depth = new_model.layers[layer_ind].kernel_size[2]
                    new_model.layers[layer_ind].set_weights(
                        [np.tile(np.expand_dims(old_model.layers[layer_ind - (4 * num_mods - 2)].get_weights()[0], axis=2) / num_depth,
                                 (1, 1, num_depth, 1, 1))])
                else:
                    new_model.layers[layer_ind].set_weights(
                        old_model.layers[layer_ind - (4 * num_mods - 2)].get_weights())
            except:
                ind_weights = old_model.layers[layer_ind - (4 * num_mods - 2)].get_weights()
                if len(ind_weights) == 1:
                    try:
                        if two_or_three_dim == 3:
                            num_depth = new_model.layers[layer_ind].kernel_size[2]
                            new_model.layers[layer_ind].set_weights(
                                [np.tile(np.expand_dims(ind_weights[0], axis=2) / (2.*num_depth),
                                         (1, 1, num_depth, 2, 1))])
                        else:
                            new_model.layers[layer_ind].set_weights(
                                [np.tile(ind_weights[0] / 2., (1, 1, 2, 1))])
                    except:
                        ind_mult = 0
                        while True:
                            ind_mult += 1
                            try:
                                if two_or_three_dim == 3:
                                    num_depth = new_model.layers[layer_ind].kernel_size[2]
                                    new_model.layers[layer_ind].set_weights(
                                        [np.concatenate(
                                            (np.tile(np.expand_dims(ind_weights[0][:, :, :-ind_mult * 32, :], axis=2) / (2. * num_depth),
                                                     (1, 1, num_depth, 2, 1)),
                                             np.expand_dims(ind_weights[0][:, :, -ind_mult * 32:, :], axis=2)), axis=3)])
                                else:
                                    new_model.layers[layer_ind].set_weights(
                                        [np.concatenate(
                                            (np.tile(ind_weights[0][:, :, :-ind_mult * 32, :] / 2., (1, 1, 2, 1)),
                                             ind_weights[0][:, :, -ind_mult * 32:, :]), axis=2)])
                                break
                            except:
                                if ind_mult > 10000:
                                    raise Exception('Unable to transfer weights properly')
                else:
                    try:
                        temp_weights_list = list()
                        for n in range(len(ind_weights)):
                            temp_weights_list.append(np.tile(ind_weights[n], (2,)))
                        new_model.layers[layer_ind].set_weights(temp_weights_list)
                    except:
                        ind_mult = 0
                        while True:
                            ind_mult += 1
                            temp_weights_list = list()
                            try:
                                for n in range(len(ind_weights)):
                                    temp_weights_list.append(np.concatenate((
                                                                            np.tile(ind_weights[n][:-ind_mult * 32],
                                                                                    (2,)),
                                                                            ind_weights[n][-ind_mult * 32:])))
                                new_model.layers[layer_ind].set_weights(temp_weights_list)
                                break
                            except:
                                if ind_mult > 10000:
                                    raise Exception('Unable to transfer weights properly')


    return new_model

