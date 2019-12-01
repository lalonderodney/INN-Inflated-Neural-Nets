'''
INN: Inflated Neural Networks for IPMN Diagnosis
Original Paper by Rodney LaLonde, Irene Tanner, Katerina Nikiforaki, Georgios Z. Papadakis, Pujan Kandel,
Candice W. Bolan, Michael B. Wallace, Ulas Bagci
(https://link.springer.com/chapter/10.1007/978-3-030-32254-0_12, https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for training models. Please see the README for details about training.
'''

from __future__ import print_function, division

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from keras.optimizers import Adam
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.metrics import binary_accuracy, categorical_accuracy
import tensorflow as tf

from custom_losses import binary_crossentropy_loss
from load_3D_data import load_class_weights, generate_train_batches, generate_val_batches

debug = True

def get_loss(training_list, net, choice):
    # Weighted binary cross-entropy loss
    if choice == 'w_bce':
        loss = binary_crossentropy_loss()
        class_weights = load_class_weights(train_list=training_list)
    # Binary cross-entropy loss
    elif choice == 'bce':
        loss = binary_crossentropy_loss()
        class_weights = None
    # Cross-entropy loss
    elif choice == 'w_ce':
        loss = 'categorical_crossentropy'
        class_weights = load_class_weights(train_list=training_list)
    # Weighted cross-entropy loss
    elif choice == 'ce':
        loss = 'categorical_crossentropy'
        class_weights = None
    else:
        raise Exception("Unknown loss_type.")

    return loss, class_weights

def get_callbacks(arguments):
    # Callback function for TF/Keras, csv log, tboard, checkpoint, lr_reduce, and early_stop
    if arguments.num_classes == 2:
        out_num = 'binary_'
    else:
        out_num = 'categorical_'
    monitor_name = 'val_' + out_num + 'accuracy'

    csv_logger = CSVLogger(os.path.join(arguments.log_dir, arguments.output_name + '_log_' + arguments.time + '.csv'), separator=',')
    tb = TensorBoard(arguments.tf_log_dir, histogram_freq=0)
    model_checkpoint = ModelCheckpoint(os.path.join(arguments.check_dir,
                                            arguments.output_name + '_model_' + arguments.time + '.hdf5'),
                                       monitor=monitor_name, save_best_only=True, save_weights_only=True,
                                       verbose=1, mode='max')
    lr_reducer = ReduceLROnPlateau(monitor=monitor_name, factor=0.05, cooldown=0, patience=10,verbose=1, mode='max')
    early_stopper = EarlyStopping(monitor=monitor_name, min_delta=0.001, patience=21, verbose=0, mode='max')

    return [model_checkpoint, csv_logger, lr_reducer, early_stopper, tb]

def compile_model(args, train_list, net_input_shape, uncomp_model):
    # Set optimizer to Adam
    try:
        opt = Adam(lr=args.initial_lr, beta_1=0.99, beta_2=0.999, decay=1e-6, amsgrad=True)
    except:
        opt = Adam(lr=args.initial_lr, beta_1=0.99, beta_2=0.999, decay=1e-6)

    # A set of useful metrics
    metrics = [tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(curve='PR'),
               tf.keras.metrics.AUC(curve='ROC')]

    if args.num_classes > 2:
        metrics.append(categorical_accuracy)
    else:
        metrics.append(binary_accuracy)

    # Get the loss function and weights
    loss, loss_weighting = get_loss(training_list=train_list, net=args.net, choice=args.loss)

    # If using CPU or single GPU, compile the model with the chosen loss, optimizer, and metrics
    if args.gpus <= 1:
        uncomp_model.compile(optimizer=opt, loss=loss, metrics=metrics)
        return uncomp_model, loss_weighting
    # If using multiple GPUs, compile the model with the chosen loss, optimizer, and metrics
    else:
        from keras.utils.training_utils import multi_gpu_model
        with tf.device("/cpu:0"):
            uncomp_model.compile(optimizer=opt, loss=loss, metrics=metrics)
            model = multi_gpu_model(uncomp_model, gpus=args.gpus)
            model.__setattr__('callback_model', uncomp_model)
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
        return model, loss_weighting


def plot_training(training_history, network, n_classes, out_dir, out_name, exp_time):
    # Basic plotting function the plots the training history
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))
    f.suptitle(network, fontsize=18)

    ax1.plot(training_history.history['precision'])
    ax1.plot(training_history.history['recall'])
    ax1.plot(training_history.history['auc_roc'])
    ax1.plot(training_history.history['auc_pr'])
    if n_classes == 2:
        ax1.plot(training_history.history['binary_accuracy'])
    else:
        ax1.plot(training_history.history['categorical_accuracy'])
    ax1.plot(training_history.history['val_precision'])
    ax1.plot(training_history.history['val_recall'])
    ax1.plot(training_history.history['val_auc_roc'])
    ax1.plot(training_history.history['val_auc_pr'])
    if n_classes == 2:
        ax1.plot(training_history.history['val_binary_accuracy'])
    else:
        ax1.plot(training_history.history['val_categorical_accuracy'])

    ax1.set_title('Precision, Recall, AUC, and Accuracy')
    ax1.legend(['Train_Precision', 'Train_Recall', 'Train_AUC_ROC', 'Train_AUC_PR', 'Train_Accuracy', 'Val_Precision',
                'Val_Recall', 'Val_AUC_ROC', 'Val_AUC_PR', 'Val_Accuracy'],
               loc='lower right')
    ax1.set_yticks(np.arange(0, 1.05, 0.05))
    ax1.set_xticks(np.arange(start=0, stop=len(training_history.history['precision']),
                                    step=int(np.ceil(len(training_history.history['precision'])/10))))
    ax1.grid(True)
    gridlines1 = ax1.get_xgridlines() + ax1.get_ygridlines()
    for line in gridlines1:
        line.set_linestyle('-.')

    ax2.plot(training_history.history['loss'])
    ax2.plot(training_history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(['Train', 'Val'], loc='upper right')
    ax1.set_xticks(np.arange(start=0, stop=len(training_history.history['loss']),
                             step=int(np.ceil(len(training_history.history['loss']) / 10))))
    ax2.grid(True)
    gridlines2 = ax2.get_xgridlines() + ax2.get_ygridlines()
    for line in gridlines2:
        line.set_linestyle('-.')

    f.savefig(os.path.join(out_dir, out_name + '_plots_' + exp_time + '.png'))
    plt.close()

def train(args, train_list, val_list, u_model, net_input_shape):
    # Compile the loaded model
    model, loss_weights = compile_model(args=args, train_list=train_list, net_input_shape=net_input_shape,
                                        uncomp_model=u_model)

    # Load pre-trained weights
    if args.custom_weights_path != '':
        try:
            model.load_weights(args.custom_weights_path)
        except Exception as e:
            print(e)
            print('!!! Failed to load weights file. Training without pre-training weights. !!!')

    # Set the callbacks
    callbacks = get_callbacks(args)

    # Training the network
    history = model.fit_generator(
        generate_train_batches(root_path=args.data_root_dir, train_list=train_list, net_shape=net_input_shape,
                               mod_dirs=args.modality_dir_list, exp_name=args.exp_name, net=args.net,
                               MIP_choices=args.MIP_choices, n_class=args.num_classes, batchSize=args.batch_size,
                               numSlices=args.slices, subSampAmt=args.subsamp, stride=args.stride,
                               shuff=args.shuffle_data, aug_data=args.aug_data),
        max_queue_size=40, workers=4, use_multiprocessing=False,
        steps_per_epoch=int(np.ceil(len(train_list)/args.batch_size*12)), # 12 avg. num of loops in train generator
        validation_data=generate_val_batches(root_path=args.data_root_dir, val_list=val_list, net_shape=net_input_shape,
                                             mod_dirs=args.modality_dir_list, exp_name=args.exp_name, net=args.net,
                                             MIP_choices=args.MIP_choices, n_class=args.num_classes,
                                             batchSize=args.batch_size, numSlices=args.slices, subSampAmt=0,
                                             stride=args.stride, shuff=args.shuffle_data),
        validation_steps=int(np.ceil(len(val_list)/args.batch_size)),
        epochs=args.epochs, class_weight=loss_weights, callbacks=callbacks, verbose=args.verbose)

    # Plot the training data collected
    plot_training(history, args.net, args.num_classes, args.output_dir, args.output_name, args.time)
