'''
INN: Inflated Neural Networks for IPMN Diagnosis
Original Paper by Rodney LaLonde, Irene Tanner, Katerina Nikiforaki, Georgios Z. Papadakis, Pujan Kandel,
Candice W. Bolan, Michael B. Wallace, Ulas Bagci
(https://link.springer.com/chapter/10.1007/978-3-030-32254-0_12, https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This is the main file for the project. From here you can train and test the models.
Please see the README for detailed instructions for this project.
'''

from __future__ import print_function

import os
import argparse
import csv
from time import gmtime, strftime
time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

from tqdm import tqdm
import SimpleITK as sitk

from load_3D_data import load_data, convert_data_to_numpy
from model_helper import create_model


def main(args):
    # Ensure training, testing, and manip are not all turned off
    assert (args.train or args.test or args.pred), \
        'Cannot have train, test, and pred all set to 0, Nothing to do.'

    # Set the output name to the experiment name, testing split, and all relevant hyperparameters for easy reference.
    args.output_name = 'exp-{}_split-{}_batch-{}_shuff-{}_aug-{}_loss-{}_slic-{}_sub-{}_strid-{}_lr-{}_' \
                       'dpre-{}_fbase-{}_dis-{}'.format(
                        args.exp_name, args.split_num, args.batch_size, args.shuffle_data, args.aug_data, args.loss,
                        args.slices, args.subsamp, args.stride, args.initial_lr, args.use_default_pretrained,
                        args.freeze_base_weights, args.disentangled)
    args.time = time

    # Set the name for the fusion strategy
    if args.disentangled:
        fusion_type = 'disen'
    else:
        fusion_type = 'early'

    # Create the directory for saving model checkpoints
    args.check_dir = os.path.join(args.data_root_dir, 'saved_models', args.exp_name, args.net + '_' + fusion_type)
    try:
        os.makedirs(args.check_dir)
    except:
        pass

    # Create the directory for saving csv logs
    args.log_dir = os.path.join(args.data_root_dir, 'logs', args.exp_name, args.net + '_' + fusion_type)
    try:
        os.makedirs(args.log_dir)
    except:
        pass

    # Create the directory for saving TF logs
    args.tf_log_dir = os.path.join(args.log_dir, 'tf_logs')
    try:
        os.makedirs(args.tf_log_dir)
    except:
        pass

    # Create the directory for saving train/test plots
    args.output_dir = os.path.join(args.data_root_dir, 'plots', args.exp_name, args.net + '_' + fusion_type, args.time)
    try:
        os.makedirs(args.output_dir)
    except:
        pass

    # Load images for this split
    all_imgs_list = []
    if (args.train or args.test):
        train_list, val_list, test_list = load_data(root=args.data_root_dir, mod_dirs=args.modality_dir_list,
                                                    exp_name=args.exp_name, split=args.split_num,
                                                    k_folds=args.k_fold_splits, val_split=args.val_split,
                                                    rand_seed=args.rand_seed)

        # Print the images selected for validation
        all_imgs_list = all_imgs_list + list(train_list) + list(val_list) + list(test_list)
        print('\nFound a total of {} images with lables.'.format(len(all_imgs_list)))
        print('\t{} images for training.'.format(len(train_list)))
        print('\t{} images for validation.'.format(len(val_list)))
        print('\t{} images for testing.'.format(len(test_list)))
        print('\nRandomly selected validation images:')
        print(val_list)
        print('\n')

    # If the user selected to do predictions (no GT), load those prediction images
    if args.pred:
        with open(os.path.join(args.data_root_dir, 'split_lists', args.exp_name,
                               'pred_split_{}.csv'.format(args.split_num)), 'rb') as f:
            reader = csv.reader(f)
            pred_list = list(reader)
        all_imgs_list = all_imgs_list + list(pred_list)

    # This creates all images up front instead of dynamically during training. Beneficial if paying for GPU hours.
    if args.create_all_imgs:
        print('-' * 98, '\nCreating all images... This will take some time.\n', '-' * 98)
        for img_pairs in tqdm(all_imgs_list):
            _, _ = convert_data_to_numpy(root_path=args.data_root_dir, img_names=img_pairs,
                                         mod_dirs=args.modality_dir_list, exp_name=args.exp_name,
                                         no_masks=False, overwrite=True)

    # Set the network input shape depending on using a 2D or 3D network.
    if args.net.find('3d') != -1 or args.net.find('inflated') != -1:
        net_input_shape = [None, None, args.slices, 3 * (len(args.modality_dir_list.split(',')) - 1)]
    else:
        net_input_shape = [None, None, 3 * (len(args.modality_dir_list.split(',')) - 1)]

    # Create the model for training/testing/manipulation
    train_shape = sitk.ReadImage(os.path.join(args.data_root_dir, all_imgs_list[0][0])).GetSize()[:-1]
    args.resize_shape = [args.resize_hei, args.resize_wid]
    if args.resize_shape[0] is not None:
        train_shape[0] = args.resize_shape[0]
    if args.resize_shape[1] is not None:
        train_shape[1] = args.resize_shape[1]

    # Update the network input shape based on our data size
    train_shape = (train_shape[0] // (2 ** 6) * (2 ** 6), train_shape[1] // (2 ** 6) * (2 ** 6))  # Assume 6 downsamples
    net_input_shape[0] = train_shape[0]
    net_input_shape[1] = train_shape[1]
    model = create_model(args=args, input_shape=net_input_shape)

    # Print the model summary (try except is for Keras version compatibility)
    try:
        from keras.utils import print_summary
        print_summary(model=model, positions=[.38, .69, .8, 1.])
    except:
        model.summary()

    if args.train:
        # Run training
        print('-'*98,'\nRunning Training\n','-'*98)
        from train_class import train
        train(args, train_list, val_list, model, net_input_shape)

    if args.test:
        # Run testing
        print('-'*98,'\nRunning Testing\n','-'*98)
        from test_class import test
        test(args, test_list, model, net_input_shape)

    if args.pred:
        # Run prediction on new data
        print('-'*98,'\nRunning Prediction\n','-'*98)
        from predict_class import predict
        predict(args, pred_list, model, net_input_shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train INN on Medical Data')
    # What you want to run arguments
    parser.add_argument('--train', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable training.')
    parser.add_argument('--test', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable testing.')
    parser.add_argument('--pred', type=int, default=0, choices=[0,1],
                        help='Set to 1 to enable prediction.')

    # Data related arguments
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='The root directory for your data.')
    parser.add_argument('--exp_name', type=str, default='NvsLRvsHR',
                        help='Name for experiment.')
    parser.add_argument('--save_prefix', type=str, default='',
                        help='Prefix to append to saved CSV.')
    parser.add_argument('--modality_dir_list', type=str, required=True,
                        help='A comma separated list of the different modality image folders (including the ground '
                             'truth mask, put last in list), relative to data_root_dir.'
                             'E.g. "Resized_T1_Registered, Resized_T2, Resized_Masks"')
    parser.add_argument('--create_all_imgs', type=int, default=0, choices=[0,1],
                        help='Set to 1 to make all images up front (otherwise they will be created as needed).')

    # Experiment related arguments
    parser.add_argument('--k_fold_splits', type=int, default=10,
                        help='Number of training splits to create for k-fold cross-validation.')
    parser.add_argument('--split_num', type=str, default='0',
                        help='Which training split to train/test on.')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes for classification.')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Percentage between 0 and 1 of training split to use as validation.')
    parser.add_argument('--rand_seed', type=int, default=5,
                        help='Random seed for training splits.')

    # Network related arguments
    parser.add_argument('--net', type=str.lower, default='inflated_inceptionv3',
                        choices=['inceptionv3', 'inflated_inceptionv3',
                                 'densenet121', 'inflated_densenet121'],
                        help='Choose your network.')
    parser.add_argument('--disentangled', type=int, default=0, choices=[0,1],
                        help='Set to 1 to use a disentangled modality approach, set to 0 to use early fusion.')
    parser.add_argument('--MIP_choices', type=str, default='0,1', required=False,
                        help='ONLY IF 2D NET IS CHOSEN, A comma separated list of equal length to the '
                             'modality_dir_list - 1 (we don\'t need for the mask). '
                             '0: Minimum Intensity Projection, 1: Maximum Intensity Projection.')
    parser.add_argument('--use_default_pretrained', type=int, default=0, choices=[0,1],
                        help='Set to 1 to enable using pre-trained ImageNet weights.')
    parser.add_argument('--freeze_base_weights', type=int, default=0, choices=[0,1],
                        help='Set to 1 to freeze the pre-trained weights.')
    parser.add_argument('--loss', type=str.lower, default='w_ce', choices=['bce', 'w_bce', 'ce', 'w_ce'],
                        help='Which loss to use. "bce" and "w_bce": unweighted and weighted binary cross entropy'
                             '"dice": soft dice coefficient, "mar": margin loss.')
    parser.add_argument('--custom_weights_path', type=str, default='',
                        help='full/path/to/trained_model.hdf5. Set to "" for none.')
    parser.add_argument('--test_weights_path', type=str, default='',
                        help='full/path/to/trained_model.hdf5. Set to "" if testing immediately after training, and '
                             'the trained weights from training will be used.')

    # Data shape and sampling related arguments
    parser.add_argument('--form_batches', type=str.lower, default='resize_avg',
                        choices=['resize_max', 'resize_min', 'resize_std', 'resize_avg', 'crop_min'],
                        help='To form batches for mini-batch training, all samples in a batch must be the same size. '
                             'When differences occur choose one of these options... '
                             '    resize_max: resize all images to the largest width and height values.'
                             '    resize_min: resize all images to the smallest width and height values. '
                             '    resize_std: resize all images to a standard size, specify --resize_hei --resize_wid.'
                             '    resize_avg: resize all images to the average width and height values.'
                             '    crop_min: crop images using random crop function to smallest height and width values.')
    parser.add_argument('--slices', type=int, default=5, choices=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31],
                        help='Number of slices to include for training/testing. '
                             'For 2D Classification, it is the number of slices to use for MIP: h x w x modalities.'
                             'For 3D classification and segmentation it is the number of slices input:'
                             'h x w x modalities x slices.')
    parser.add_argument('--subsamp', type=int, default=0,
                        help='Number of slices to skip when forming 3D samples for training. Enter -1 for random '
                             'subsampling up to 5 percent of total slices.')
    parser.add_argument('--stride', type=int, default=1,
                        help='Number of slices to move when generating the next sample.')
    parser.add_argument('--crop_hei', type=int, default=None,
                        help="Random image crop height for training")
    parser.add_argument('--crop_wid', type=int, default=None,
                        help="Random image crop width for training")
    parser.add_argument('--resize_hei', type=int, default=None,
                        help="Image resize height for forming equal size batches")
    parser.add_argument('--resize_wid', type=int, default=None,
                        help="Image resize width for forming equal size batches")

    # Batch_size, learning rate, epochs, augmentation, shuffle.
    parser.add_argument('--shuffle_data', type=int, default=1, choices=[0,1],
                        help='Whether or not to shuffle the training data (both per epoch and in slice order.')
    parser.add_argument('--aug_data', type=int, default=1, choices=[0,1],
                        help='Whether or not to use data augmentation during training.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training/testing.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train for, not accounting for early stopping.')
    parser.add_argument('--initial_lr', type=float, default=0.0001,
                        help='Initial learning rate for Adam.')

    # DenseNet related arguments
    parser.add_argument('--reduction', type=float, default=0.5, help='Reduction rate for DenseNet.')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight Decay.')
    parser.add_argument('--group_norm', type=int, default=1e-4, help='Group Norm.')

    # Output to user
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help='Set the verbose value for training. 0: Silent, 1: per iteration, 2: per epoch.')

    # GPU related arguments
    parser.add_argument('--which_gpus', type=str, default="0",
                        help='Enter "-2" for CPU only, "-1" for all GPUs available, '
                             'or a comma separated list of GPU id numbers ex: "0,1,4".')
    parser.add_argument('--gpus', type=int, default=-1,
                        help='Number of GPUs you have available for training. '
                             'If entering specific GPU ids under the --which_gpus arg or if using CPU, '
                             'then this number will be inferred, else this argument must be included.')
    arguments = parser.parse_args()


    # Mask out the GPUs if the user selects CPU only (-2).
    if arguments.which_gpus == -2:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # If the user selects to use all GPUs, must specify how many GPUs there are with --gpus
    elif arguments.which_gpus == '-1':
        assert (arguments.gpus != -1), 'Use all GPUs option selected under --which_gpus, with this option the user MUST ' \
                                  'specify the number of GPUs available with the --gpus option.'
    # This is the preferred default, just a comma-separated list of GPU IDs.
    else:
        arguments.gpus = len(arguments.which_gpus.split(','))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments.which_gpus)

    if arguments.gpus > 1:
        assert arguments.batch_size >= arguments.gpus, 'Error: Must have at least as many items per batch as GPUs ' \
                                                       'for multi-GPU training. For model parallelism instead of ' \
                                                       'data parallelism, modifications must be made to the code.'

    main(arguments)
