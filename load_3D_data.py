'''
INN: Inflated Neural Networks for IPMN Diagnosis
Original Paper by Rodney LaLonde, Irene Tanner, Katerina Nikiforaki, Georgios Z. Papadakis, Pujan Kandel,
Candice W. Bolan, Michael B. Wallace, Ulas Bagci
(https://link.springer.com/chapter/10.1007/978-3-030-32254-0_12, https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for loading training, validation, and testing data into the models.
It is specifically designed to handle 3D single-channel medical data.
Modifications will be needed to train/test on normal 3-channel images.
'''

from __future__ import print_function, division

import threading
import os
import csv
from glob import glob

import numpy as np
from numpy.random import rand, shuffle
import SimpleITK as sitk
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from skimage.measure import find_contours
from scipy.interpolate import interp1d
from tqdm import tqdm
from keras.preprocessing.image import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from custom_data_aug import elastic_transform, salt_pepper_noise

# GLOBAL VARIABLES FOR NORMALIZATION
i_min=1; i_max=99; i_s_min=1; i_s_max=100; l_percentile=10; u_percentile=90; step=10
percs = np.concatenate(([i_min], np.arange(l_percentile, u_percentile+1, step), [i_max]))
standard_scale = np.zeros((2,len(percs))) # TODO: Don't hardcode 2 modalities

debug = False

def load_data(root, mod_dirs, exp_name, split=0, k_folds=4, val_split=0.1, rand_seed=5):
    # Main functionality of loading and spliting the data
    def _load_data():
        with open(os.path.join(root, 'split_lists', exp_name, 'train_split_{}.csv'.format(split)), 'rb') as f:
            reader = csv.reader(f)
            training_list = list(reader)
        with open(os.path.join(root, 'split_lists', exp_name, 'test_split_{}.csv'.format(split)), 'rb') as f:
            reader = csv.reader(f)
            test_list = list(reader)
        X = np.asarray(training_list)[:,:-1]
        orig_data = [x[0].split(os.sep)[1] for x in X]
        y = np.asarray(training_list)[:, -1].astype(int)
        uniq_data = list()
        uniq_label = list()
        map_data = list()
        i = 0
        for n, x in enumerate(orig_data):
            if x not in uniq_data:
                uniq_data.append(x)
                uniq_label.append(y[n])
                map_data.append(i)
                i += 1
            else:
                map_data.append(uniq_data.index(x))

        map_data = np.asarray(map_data)

        X_train, X_val, y_train, y_val = train_test_split(uniq_data, uniq_label, test_size=val_split, random_state=12,
                                                          stratify=uniq_label)

        full_X_train = list()
        full_y_train = list()
        for x in X_train:
            map_val = uniq_data.index(x)
            full_X_train.extend(X[map_data == map_val])
            full_y_train.extend(y[map_data == map_val])
        full_X_val = list()
        full_y_val = list()
        for x in X_val:
            map_val = uniq_data.index(x)
            full_X_val.extend(X[map_data == map_val])
            full_y_val.extend(y[map_data == map_val])

        new_train_list = np.concatenate((full_X_train, np.expand_dims(full_y_train, axis=1)), axis=1)
        val_list = np.concatenate((full_X_val, np.expand_dims(full_y_val, axis=1)), axis=1)
        return new_train_list, val_list, test_list

    # Try-catch to handle calling split data before load only if files are not found.
    try:
        new_training_list, validation_list, testing_list = _load_data()
        return new_training_list, validation_list, testing_list
    except:
        # Create the training and test splits if not found
        split_data(root, mod_dirs, exp_name, num_splits=k_folds, rand_seed=rand_seed)
        try:
            new_training_list, validation_list, testing_list = _load_data()
            return new_training_list, validation_list, testing_list
        except Exception as e:
            print(e)
            print('Failed to load data, see load_data in load_3D_data.py')
            exit(1)


def split_data(root_path, mod_dirs_paths, exp_name, num_splits=4, rand_seed=5):
    mod_dirs_list = mod_dirs_paths.split(',')
    # All modalities must name img_dirs the same, otherwise cannot know how to match them
    img_dirs_list = sorted(glob(os.path.join(root_path, mod_dirs_list[0].strip(), '*')))

    # Load the GT labels for IPMN
    IPMN_GT = dict()
    with open(os.path.join(root_path, 'IPMN_Ground_Truth.csv'), 'rb') as f:
        for k, v in csv.reader(f):

            IPMN_GT[k] = v

    img_dirs_pairs_list = []
    for img_dir in img_dirs_list:
        imgs_all_mods = []
        for mod_dir in mod_dirs_list:
            imgs_per_mod = []
            for ext in ('*.mhd', '*.hdr', '*.nii'):
                # NOTE: If more than one file is present in the CAD folder...
                # MUST have matching prefix to guarantee sorted will match them correctly.
                img_path_list = sorted(glob(os.path.join(root_path, mod_dir.strip(), os.path.basename(img_dir), ext)))
                imgs_per_mod.extend(img_path_list)
            imgs_all_mods.append(imgs_per_mod)
        if len(imgs_all_mods) == len(mod_dirs_list):
            try:
                imgs_all_mods.append(IPMN_GT[os.path.basename(img_dir)])
            except:
                print('Unable to load GT pathology for {}: \nSetting to -1!'.format(os.path.basename(img_dir)))
                imgs_all_mods.append('-1')

            if int(imgs_all_mods[-1]) == 3:
                imgs_all_mods[-1] = '2' # Lump class 3 in with class 2

            if (int(imgs_all_mods[-1]) == 0 or int(imgs_all_mods[-1]) == 1 or int(imgs_all_mods[-1]) == 2):
                img_dirs_pairs_list.append(imgs_all_mods)
    assert len(img_dirs_pairs_list) != 0, 'Unable to find any files. Check split_data function.'

    outdir = os.path.join(root_path,'split_lists', exp_name)
    try:
        os.makedirs(outdir)
    except:
        pass

    final_img_list = list(np.array(img_dirs_pairs_list)[:,:-1])
    final_label_list = list(np.array(img_dirs_pairs_list)[:,-1].astype(int))

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=rand_seed)
    n = 0
    for train_index, test_index in skf.split(final_img_list, final_label_list):
        with open(os.path.join(outdir,'train_split_{}.csv'.format(n)), 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in train_index:
                for j in range(np.asarray(img_dirs_pairs_list[i][0]).size):
                    writer.writerow([img_dirs_pairs_list[i][0][j].split(root_path)[1][1:],
                                     img_dirs_pairs_list[i][1][j].split(root_path)[1][1:],
                                     img_dirs_pairs_list[i][2][j].split(root_path)[1][1:],
                                     img_dirs_pairs_list[i][3]])
        with open(os.path.join(outdir,'test_split_{}.csv'.format(n)), 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in test_index:
                for j in range(np.asarray(img_dirs_pairs_list[i][0]).size):
                    writer.writerow([img_dirs_pairs_list[i][0][j].split(root_path)[1][1:],
                                     img_dirs_pairs_list[i][1][j].split(root_path)[1][1:],
                                     img_dirs_pairs_list[i][2][j].split(root_path)[1][1:],
                                     img_dirs_pairs_list[i][3]])
        n += 1


def compute_avg_size(root, all_data_list):
    hei_wid = list()
    for img_list in tqdm(all_data_list):
        img = sitk.ReadImage(os.path.join(root, img_list[0]))
        hei_wid.append([img.GetSize()[0], img.GetSize()[1]])
    hei_wid = np.asarray(hei_wid)
    return np.mean(hei_wid, axis=0), np.std(hei_wid, axis=0)


def compute_min_max_slices(root, all_data_list):
    min = 999999
    max = 0
    for img_list in tqdm(all_data_list):
        img = sitk.ReadImage(os.path.join(root, img_list[0]))
        slices = img.GetSize()[2]
        if slices < min:
            min = slices
        if slices > max:
            max = slices

    return min, max


def load_class_weights(train_list):
    y = np.array(train_list)[:,3].astype(int)
    class_weight_list = len(y) / (len(np.unique(y)) * np.bincount(y)).astype(np.float32)
    class_weights = dict(zip(np.unique(y), class_weight_list))
    return class_weights


def hm_scale(root_path, mod_dirs, exp_name, index, no_masks=False):
    """
    https://github.com/jcreinhold/intensity-normalization
    determine the standard scale for the set of images
    Args:
        root_path
        img_fns (list): set of NifTI MR image paths which are to be normalized
        mask_fns (list): set of corresponding masks (if not provided, estimated)
        i_min (float): minimum percentile to consider in the images
        i_max (float): maximum percentile to consider in the images
        i_s_min (float): minimum percentile on the standard scale
        i_s_max (float): maximum percentile on the standard scale
        l_percentile (int): middle percentile lower bound (e.g., for deciles 10)
        u_percentile (int): middle percentile upper bound (e.g., for deciles 90)
        step (int): step for middle percentiles (e.g., for deciles 10)
    Returns:
        standard_scale (np.ndarray): average landmark intensity for images
        percs (np.ndarray): array of all percentiles used
    """
    global i_min; global i_max; global i_s_min; global i_s_max; global l_percentile; global u_percentile; global step
    global percs; global standard_scale

    train_list, val_list, test_list = load_data(root_path, mod_dirs, exp_name)

    img_fns = list(np.concatenate((train_list, val_list, test_list), axis=0)[:, index])
    mask_fns = list(np.concatenate((train_list, val_list, test_list), axis=0)[:, -2])
    mask_fns = [None] * len(img_fns) if mask_fns is None else mask_fns

    for i, (img_fn, mask_fn) in enumerate(zip(img_fns, mask_fns)):
        img_data = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root_path, img_fn)))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root_path, mask_fn))) if mask_fn is not None else None
        try:
            mask_data = img_data > np.mean(img_data) if mask is None else mask
            masked = img_data[mask_data > 0]
        except:
            raise Exception('Shape mismatch between mask and image for file {}.'.format(img_fn))
        landmarks = np.percentile(masked, percs)
        min_p = np.percentile(masked, i_min)
        max_p = np.percentile(masked, i_max)
        f = interp1d([min_p, max_p], [i_s_min, i_s_max])
        landmarks = np.array(f(landmarks))
        standard_scale[index] += landmarks
    standard_scale[index] = standard_scale[index] / len(img_fns)
    return None


# def do_hist_norm(img, mask=None):
#     """
#     https://github.com/jcreinhold/intensity-normalization
#     do the Nyul and Udupa histogram normalization routine with a given set of learned landmarks
#     """
#     global percs; global standard_scale
#
#     mask = img > np.mean(img) if mask is None else mask
#     masked = img[mask > 0]
#     landmarks = np.percentile(masked, percs)
#     f = interp1d(landmarks, standard_scale, fill_value='extrapolate')
#     normed = np.zeros(img.shape)
#     normed[mask > 0] = f(masked)
#     return normed


def convert_data_to_numpy(root_path, img_names, mod_dirs, exp_name, no_masks=False, overwrite=False):
    global percs; global standard_scale

    fname = img_names[0].split(os.sep)[1]
    # This is a custom splitting based on who made the masks, the Greece team or Irene
    if img_names[0].split(os.sep)[2].split('_')[0] == 'greece' or img_names[0].split(os.sep)[2].split('_')[0] == 'irene':
        fname = fname + '_' + img_names[0].split(os.sep)[2].split('_')[0]
    numpy_path = os.path.join(root_path, 'np_files')
    fig_path = os.path.join(root_path, 'figs')
    try:
        os.makedirs(numpy_path)
    except:
        pass
    try:
        os.makedirs(fig_path)
    except:
        pass

    if not overwrite:
        try:
            with np.load(os.path.join(numpy_path, fname + '.npz')) as data:
                if no_masks:
                    return np.stack((data['T1'], data['T2']), axis=-1)
                else:
                    return np.stack((data['T1'], data['T2']), axis=-1), data['mask']
        except:
            pass

    try:
        corrected_imgs = []
        if not no_masks:
            f, ax = plt.subplots(len(img_names[:-2]), 4, figsize=(20, 10))
            itk_pancreas_mask = sitk.ReadImage(os.path.join(root_path, img_names[-2]))
            pancreas_mask = sitk.GetArrayFromImage(itk_pancreas_mask)
            pancreas_mask = np.rollaxis(pancreas_mask, 0, 3)
            pancreas_mask[pancreas_mask >= 0.5] = 1
            pancreas_mask[pancreas_mask != 1] = 0

            h_rem = pancreas_mask.shape[0] % 2 ** 5
            w_rem = pancreas_mask.shape[1] % 2 ** 5
            if h_rem != 0 or w_rem != 0:
                pancreas_mask = np.pad(pancreas_mask, ((int(np.ceil(h_rem / 2.)), int(np.floor(h_rem / 2.))),
                                                       (int(np.ceil(w_rem / 2.)), int(np.floor(w_rem / 2.))),
                                                       (0, 0)), 'symmetric')

            pancreas_mask = pancreas_mask.astype(np.uint8)
            first, last, largest = find_mask_endpoints(pancreas_mask)
            pancreas_contours = find_contours(pancreas_mask[:, :, largest], 0.8)
        else:
            f, ax = plt.subplots(len(img_names[:-2]), 3, figsize=(15, 10))
            largest = sitk.ReadImage(os.path.join(root_path, img_names[0])).GetSize()[-1]//2 # Just take the center slice

        for ind, img_name in enumerate(img_names[:-2]):
            itk_img = sitk.ReadImage(os.path.join(root_path, img_name))
            orig_img = sitk.GetArrayFromImage(itk_img)
            mod_name = img_name.split(os.sep)[0].split('_')[1]

            ax[ind, 0].imshow(orig_img[largest,:, :], cmap='gray')
            if not no_masks:
                for contour in pancreas_contours:
                    ax[ind, 0].plot(contour[:, 1], contour[:, 0], linewidth=2, color='r', alpha=0.5)
            ax[ind, 0].set_title('{} Original Image'.format(mod_name))
            ax[ind, 0].axis('off')

            print('\tPerforming N4BiasFieldCorrection on {} Image.'.format(mod_name))
            shrink_factor = 1
            number_fitting_levels = 4
            number_of_iterations = 50
            itk_mask = sitk.OtsuThreshold(itk_img, 0, 1, 200)
            inputImage = sitk.Shrink(itk_img, [shrink_factor] * itk_img.GetDimension())
            maskImage = sitk.Shrink(itk_mask, [shrink_factor] * itk_mask.GetDimension())
            inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations([number_of_iterations] * number_fitting_levels)
            corrected_itk_img = corrector.Execute(inputImage, maskImage)
            corrected_img = sitk.GetArrayFromImage(corrected_itk_img)

            ax[ind, 1].imshow(corrected_img[largest, :, :], cmap='gray')
            if not no_masks:
                for contour in pancreas_contours:
                    ax[ind, 1].plot(contour[:, 1], contour[:, 0], linewidth=2, color='r', alpha=0.5)
            ax[ind, 1].set_title('{} N4BiasFieldCorrected'.format(mod_name))
            ax[ind, 1].axis('off')

            print('\tPerforming CurvatureAnisotropicFilter on {} Image.'.format(mod_name))
            filtered_itk_img = sitk.CurvatureAnisotropicDiffusion(corrected_itk_img, timeStep=0.015)
            filtered_img = sitk.GetArrayFromImage(filtered_itk_img)

            ax[ind, 2].imshow(filtered_img[largest, :, :], cmap='gray')
            if not no_masks:
                for contour in pancreas_contours:
                    ax[ind, 2].plot(contour[:, 1], contour[:, 0], linewidth=2, color='r', alpha=0.5)
            ax[ind, 2].set_title('{} CurvatureAnsiotropicFiltered'.format(mod_name))
            ax[ind, 2].axis('off')

            # # Nyul and Udupa histogram normalization routine with a given set of learned landmarks
            # print('\tPerforming Nyul and Udupa histogram normalization on {} Images.'.format(mod_name))
            # if np.array_equal(standard_scale[ind], np.zeros(len(percs))):
            #     hm_scale(root_path, mod_dirs, exp_name, index=ind, no_masks=True)
            #
            # mask = sitk.GetArrayFromImage(maskImage)
            # masked = filtered_img[mask > 0]
            # landmarks = np.percentile(masked, percs)
            # f = interp1d(landmarks, standard_scale[ind], fill_value='extrapolate')
            # normed_img = f(filtered_img)

            out_img = np.rollaxis(filtered_img, 0, 3)
            out_img = out_img.astype(np.float32)
            # top_ninety = np.percentile(out_img, 99)
            # bottom_ten = np.percentile(out_img, 1)
            # out_img[out_img > top_ninety] = top_ninety
            # out_img[out_img < bottom_ten] = bottom_ten
            out_img -= np.min(out_img)
            out_img /= np.max(out_img)

            h_rem = out_img.shape[0] % 2 ** 5
            w_rem = out_img.shape[1] % 2 ** 5
            if h_rem != 0 or w_rem != 0:
                out_img = np.pad(out_img, ((int(np.ceil(h_rem / 2.)), int(np.floor(h_rem / 2.))),
                                   (int(np.ceil(w_rem / 2.)), int(np.floor(w_rem / 2.))),
                                   (0, 0)), 'symmetric')
            corrected_imgs.append(out_img)

            # ax[ind, 3].imshow(out_img[:, :, largest], cmap='gray')
            # if not no_masks:
            #     for contour in pancreas_contours:
            #         ax[ind, 3].plot(contour[:, 1], contour[:, 0], linewidth=2, color='r', alpha=0.5)
            # ax[ind, 3].set_title('{} Nyul&Udupa HistNorm'.format(mod_name))
            # ax[ind, 3].axis('off')

            # Performing MIP for plotting only
            if not no_masks:
                if out_img.shape[-1] >= 5:
                    try:
                        if mod_name == 'T1': # TODO: don't harcode this, change to args.MIP_choices
                            ax[ind, 3].imshow(np.min(out_img[:, :, largest-2:largest+2], axis=-1), cmap='gray')
                            ax[ind, 3].set_title('{} Min Intensity Projection - 5 slices'.format(mod_name))
                        elif mod_name == 'T2': # TODO: don't harcode this, change to args.MIP_choices
                            ax[ind, 3].imshow(np.max(out_img[:, :, largest-2:largest+2], axis=-1), cmap='gray')
                            ax[ind, 3].set_title('{} Max Intensity Projection - 5 slices'.format(mod_name))
                        for contour in pancreas_contours:
                            ax[ind, 3].plot(contour[:, 1], contour[:, 0], linewidth=2, color='r', alpha=0.5)
                        ax[ind, 3].axis('off')
                    except:
                        if mod_name == 'T1': # TODO: don't harcode this, change to args.MIP_choices
                            ax[ind, 3].imshow(np.min(out_img[:, :, 0:5], axis=-1), cmap='gray')
                            ax[ind, 3].set_title('{} Min Intensity Projection - 5 slices'.format(mod_name))
                        elif mod_name == 'T2': # TODO: don't harcode this, change to args.MIP_choices
                            ax[ind, 3].imshow(np.max(out_img[:, :, 0:5], axis=-1), cmap='gray')
                            ax[ind, 3].set_title('{} Max Intensity Projection - 5 slices'.format(mod_name))
                        for contour in pancreas_contours:
                            ax[ind, 3].plot(contour[:, 1], contour[:, 0], linewidth=2, color='r', alpha=0.5)
                        ax[ind, 3].axis('off')
                else:
                    try:
                        if mod_name == 'T1': # TODO: don't harcode this, change to args.MIP_choices
                            ax[ind, 3].imshow(np.min(out_img[:, :, largest-1:largest+1], axis=-1), cmap='gray')
                            ax[ind, 3].set_title('{} Min Intensity Projection - 3 slices'.format(mod_name))
                        elif mod_name == 'T2': # TODO: don't harcode this, change to args.MIP_choices
                            ax[ind, 3].imshow(np.max(out_img[:, :, largest-1:largest+1], axis=-1), cmap='gray')
                            ax[ind, 3].set_title('{} Max Intensity Projection - 3 slices'.format(mod_name))
                        for contour in pancreas_contours:
                            ax[ind, 3].plot(contour[:, 1], contour[:, 0], linewidth=2, color='r', alpha=0.5)
                        ax[ind, 3].axis('off')
                    except:
                        if mod_name == 'T1': # TODO: don't harcode this, change to args.MIP_choices
                            ax[ind, 3].imshow(np.min(out_img[:, :, 0:3], axis=-1), cmap='gray')
                            ax[ind, 3].set_title('{} Min Intensity Projection - 3 slices'.format(mod_name))
                        elif mod_name == 'T2': # TODO: don't harcode this, change to args.MIP_choices
                            ax[ind, 3].imshow(np.max(out_img[:, :, 0:3], axis=-1), cmap='gray')
                            ax[ind, 3].set_title('{} Max Intensity Projection - 3 slices'.format(mod_name))
                        for contour in pancreas_contours:
                            ax[ind, 3].plot(contour[:, 1], contour[:, 0], linewidth=2, color='r', alpha=0.5)
                        ax[ind, 3].axis('off')

        fig = plt.gcf()
        fig.suptitle(t='{}, IPMN Score: {}'.format(fname, img_names[-1]), y=0.94, fontsize=16)
        plt.savefig(os.path.join(fig_path, fname + '.png'), format='png', bbox_inches='tight')
        plt.close(fig)

        # TODO: Make this handle any number of modalities
        if not no_masks:
            np.savez_compressed(os.path.join(numpy_path, fname + '.npz'), T1=corrected_imgs[0], T2=corrected_imgs[1],
                                mask=pancreas_mask)
        else:
            np.savez_compressed(os.path.join(numpy_path, fname + '.npz'), T1=corrected_imgs[0], T2=corrected_imgs[1])

        if not no_masks:
            return np.stack((corrected_imgs[0], corrected_imgs[1]), axis=-1), pancreas_mask
        else:
            return np.stack((corrected_imgs[0], corrected_imgs[1]), axis=-1)

    except Exception as e:
        print('\n'+'-'*100)
        print('Unable to load img or masks for {}'.format(fname))
        print(e)
        print('Skipping file')
        print('-'*100+'\n')

        return np.zeros(1), np.zeros(1)

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def augmentImages(batch_of_images, batch_of_masks=None):
    # Data augmentation for deep learning, pretty standard plus two custom ones.
    for i in range(len(batch_of_images)):
        if batch_of_images.ndim == 5:
            _, h, w, c, m = batch_of_images.shape
            imgs_reshaped = np.reshape(batch_of_images[i, :, :, :, :], (h, w, c * m))
            if batch_of_masks is not None:
                mask_reshaped = np.reshape(batch_of_masks[i, :, :, :, :], (h, w, c * 1))
        else:
            imgs_reshaped = batch_of_images[i, :, :, :]
            if batch_of_masks is not None:
                mask_reshaped = batch_of_masks[i, :, :, :]

        if batch_of_masks is not None:
            img_and_mask = np.concatenate((imgs_reshaped, mask_reshaped), axis=2)
        else:
            img_and_mask = imgs_reshaped

        if np.random.randint(0,2):
            img_and_mask = random_rotation(img_and_mask, rg=30, row_axis=0, col_axis=1, channel_axis=2,
                                           fill_mode='constant', cval=0.)

        if np.random.randint(0, 2):
            img_and_mask = elastic_transform(img_and_mask, alpha=500, sigma=30, alpha_affine=1)

        if np.random.randint(0, 2):
            img_and_mask = random_shift(img_and_mask, wrg=0.1, hrg=0.1, row_axis=0, col_axis=1, channel_axis=2,
                                        fill_mode='constant', cval=0.)

        if np.random.randint(0, 2):
            img_and_mask = random_shear(img_and_mask, intensity=8, row_axis=0, col_axis=1, channel_axis=2,
                         fill_mode='constant', cval=0.)

        if np.random.randint(0, 2):
            img_and_mask = random_zoom(img_and_mask, zoom_range=(0.9, 0.9), row_axis=0, col_axis=1, channel_axis=2,
                         fill_mode='constant', cval=0.)

        if np.random.randint(0, 2):
            img_and_mask = flip_axis(img_and_mask, axis=1)

        if np.random.randint(0, 2):
            img_and_mask = flip_axis(img_and_mask, axis=0)

        if np.random.randint(0, 2):
            salt_pepper_noise(img_and_mask, salt=0.2, amount=0.04)

        if batch_of_masks is not None:
            aug_imgs = img_and_mask[:, :, :-c]
            if batch_of_images.ndim == 5:
                batch_of_masks[i, :, :, :] = np.reshape(img_and_mask[:, :, -c:], (h, w, c, 1))
            else:
                batch_of_masks[i, :, :, :] = img_and_mask[:, :, -c:]

            # Ensure the masks did not get any non-binary values.
            batch_of_masks[batch_of_masks > 0.5] = 1
            batch_of_masks[batch_of_masks <= 0.5] = 0
        else:
            aug_imgs = img_and_mask

        if batch_of_images.ndim == 5:
            batch_of_images[i, :, :, :, :] = np.reshape(aug_imgs, (h, w, c, m))
        else:
            batch_of_images[i, :, :, :] = aug_imgs

    return(batch_of_images, batch_of_masks)


''' Make the generators threadsafe in case of multiple threads '''
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def find_mask_endpoints(mask):
    first = -1
    last = -1
    for i in range(mask.shape[-1]):
        if np.any(mask[:,:,i]) and first == -1:
            first = i
        if np.any(mask[:,:,mask.shape[-1]-i-1]) and last == -1:
            last = mask.shape[-1] - i - 1
        if first != -1 and last != -1:
            break
    largest = np.argmax(np.count_nonzero(mask, axis=(0,1)))
    return first, last, largest

@threadsafe_generator
def generate_train_batches(root_path, train_list, net_shape, mod_dirs, exp_name, net, MIP_choices,
                                          n_class=1, batchSize=1, numSlices=1, subSampAmt=-1, stride=1, downSampAmt=1,
                                          shuff=1, aug_data=1):
    if n_class == 2:
        n_class = 1 # To classes is binary (0,1).

    # Create placeholders for training
    if net.find('3d') != -1 or net.find('inflated') != -1:
        img_batch = np.zeros(((batchSize, net_shape[0], net_shape[1], numSlices,
                               3*len(list(mod_dirs.split(','))[:-1]))), dtype=np.float32)
    else:
        img_batch = np.zeros(((batchSize, net_shape[0], net_shape[1], 3*len(list(mod_dirs.split(','))[:-1]))),
                             dtype=np.float32)
    gt_batch = np.zeros(((batchSize, n_class)), dtype=np.uint8)

    try:
        MIP_choices = MIP_choices.replace(' ', '')
        mip_list = MIP_choices.split(',')
    except:
        raise Exception('Unable to convert MIP_choices to a list of integers. Please check this argument.')

    while True:
        if shuff:
            shuffle(train_list)
        count = 0
        for scan_names in train_list:
            path_to_np = os.path.join(root_path,'np_files',scan_names[0].split(os.sep)[1]+'.npz')
            if scan_names[0].split(os.sep)[2].split('_')[0] == 'greece' or scan_names[0].split(os.sep)[2].split('_')[0] == 'irene':
                path_to_np = path_to_np[:-4] + '_' + scan_names[0].split(os.sep)[2].split('_')[0] + path_to_np[-4:]
            try:
                with np.load(path_to_np) as data:
                    imgs = np.stack((data['T1'], data['T2']), axis=-1) # TODO: Find modalities not hardcode
                    mask = data['mask']
            except:
                print('\nPre-made numpy array not found for {}.\n    Creating now...'.format(os.path.basename(path_to_np)))
                imgs, mask = convert_data_to_numpy(root_path=root_path, img_names=scan_names, mod_dirs=mod_dirs,
                                                       exp_name=exp_name, no_masks=False)
                if np.array_equal(imgs,np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')

            gt = int(scan_names[-1]) # GT depends on the experiment, do not save/load these, grab from train list.

            if imgs.shape[-2] < numSlices * (subSampAmt+1):
                imgs = np.pad(imgs, ((0,0), (0,0), (int(np.floor((numSlices * (subSampAmt+1) - imgs.shape[-2]) / 2)),
                                                    int(np.ceil((numSlices * (subSampAmt+1) - imgs.shape[-2]) / 2))),
                                     (0,0)), mode='symmetric')
                indicies = [0]
            else:
                mask_first_nonzero, mask_last_nonzero, mask_largest = find_mask_endpoints(mask)
                if mask_first_nonzero == -1 or mask_last_nonzero == -1 or mask_largest == -1:
                    mask_first_nonzero = 0
                    mask_last_nonzero = imgs.shape[-2]
                    mask_largest = (mask_first_nonzero + mask_last_nonzero) // 2

                if numSlices == 1:
                    subSampAmt = 0
                elif subSampAmt == -1 and numSlices > 1:
                    np.random.seed(None)
                    subSampAmt = int(rand(1) * (mask_last_nonzero - mask_first_nonzero) * 0.25)
                    while mask_last_nonzero - numSlices * (subSampAmt + 1) + 1 <= mask_first_nonzero:
                        subSampAmt -= 1
                        if subSampAmt == 0:
                            break

                indicies = np.arange(mask_first_nonzero, mask_last_nonzero - numSlices * (subSampAmt + 1) + 1, stride)
                if indicies.size == 0:
                    temp_index = mask_largest - (numSlices * (subSampAmt+1))//2
                    if temp_index >= 0 and temp_index + numSlices * (subSampAmt+1) <= imgs.shape[-2]:
                        indicies = [temp_index] # Try to guarantee at least one per scan
                    else:
                        indicies = np.arange(0, imgs.shape[-2] - numSlices * (subSampAmt + 1) + 1, stride)
                        if indicies.size == 0:
                            print('Unable to create any training examples for {}.'.format(scan_names[0]))
                        continue

            if shuff:
                shuffle(indicies)

            for j in indicies:
                if net.find('3d') != -1 or net.find('inflated') != -1:
                    for i in range(imgs.shape[-1]):
                        img_batch[count, :,:,:, 3*i:3*i+3] = \
                            np.tile(np.expand_dims(
                                imgs[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1, i], axis=-1), (1,1,1,3))
                else:
                    cropped_imgs = imgs[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1, :]
                    assert len(mip_list) == cropped_imgs.shape[-1], \
                        'Different number of MIP_choices and imaging modalities given'
                    for i, mip_choice in enumerate(mip_list):
                        if int(mip_choice) == 0:
                            img_batch[count, :, :, 3*i:3*i+3] = \
                                np.tile(np.expand_dims(np.min(cropped_imgs[:,:,:,i], axis=-1), axis=-1), (1,1,3))
                        elif int(mip_choice) == 1:
                            img_batch[count, :, :, 3*i:3*i+3] = \
                                np.tile(np.expand_dims(np.max(cropped_imgs[:,:,:,i], axis=-1), axis=-1), (1,1,3))
                        else:
                            raise Exception('Invalid choice for MIP_choices. Must be either 0 for min or 1 for max.')

                if n_class > 1:
                    gt_batch[count, :] = get_one_hot(np.asarray(int(gt)), n_class)
                else:
                    gt_batch[count, 0] = int(gt)

                count += 1
                if count % batchSize == 0:
                    count = 0
                    if aug_data:
                        img_batch, _ = augmentImages(img_batch)

                    if debug:
                        for plt_ind in range(batchSize):
                            f, ax = plt.subplots(2, figsize=(15, 10))
                            if net.find('3d') != -1 or net.find('inflated') != -1:
                                ax[0].imshow(np.squeeze(img_batch[plt_ind, :, :, 0, 0]), cmap='gray')
                            else:
                                ax[0].imshow(np.squeeze(img_batch[plt_ind, :, :, 0]), cmap='gray')
                            ax[0].set_title('T1 MIP')
                            ax[0].axis('off')
                            if net.find('3d') != -1 or net.find('inflated') != -1:
                                ax[1].imshow(np.squeeze(img_batch[plt_ind, :, :, 0, 3]), cmap='gray')
                            else:
                                ax[1].imshow(np.squeeze(img_batch[plt_ind, :, :, 3]), cmap='gray')
                            ax[1].set_title('T2 MIP')
                            ax[1].axis('off')
                            fig = plt.gcf()
                            fig.suptitle('IPMN Label: {}'.format(gt_batch[plt_ind]))
                            plt.savefig(os.path.join(root_path, 'logs', 'ex_train_{}.png'.format(plt_ind)), format='png',
                                        bbox_inches='tight')
                            plt.close(fig)

                    if img_batch.shape[3] == 1:
                        out_img_batch = np.squeeze(img_batch, axis=3)
                    else:
                        out_img_batch = img_batch
                    if net.find('caps') != -1:
                        if net.find('3d') != -1 or net.find('inflated') != -1:
                            out_recon_gt = out_img_batch[:,:,:,numSlices//2,:]
                        else:
                            out_recon_gt = out_img_batch
                        yield ([out_img_batch, gt_batch], [gt_batch, out_recon_gt])
                    else:
                        yield (out_img_batch, gt_batch)

        if count != 0:
            if aug_data:
                img_batch[:count,...], _ = augmentImages(img_batch[:count,...])
            if img_batch.shape[3] == 1:
                out_img_batch = np.squeeze(img_batch, axis=3)
            else:
                out_img_batch = img_batch
            if net.find('caps') != -1:
                if net.find('3d') != -1 or net.find('inflated') != -1:
                    out_recon_gt = out_img_batch[:,:,:,numSlices//2,:]
                else:
                    out_recon_gt = out_img_batch
                yield ([out_img_batch[:count, ...], gt_batch[:count, ...]],
                       [gt_batch[:count, ...], out_recon_gt[:count, ...]])
            else:
                yield (out_img_batch[:count, ...], gt_batch[:count, ...])

@threadsafe_generator
def generate_val_batches(root_path, val_list, net_shape, mod_dirs, exp_name, net, MIP_choices, n_class=1,
                                        batchSize=1, numSlices=1, subSampAmt=-1, stride=1, downSampAmt=1, shuff=1):
    if n_class == 2:
        n_class = 1 # To classes is binary (0,1).

    # Create placeholders for training
    if net.find('3d') != -1 or net.find('inflated') != -1:
        img_batch = np.zeros(((batchSize, net_shape[0], net_shape[1], numSlices,
                               3*len(list(mod_dirs.split(','))[:-1]))), dtype=np.float32)
    else:
        img_batch = np.zeros(((batchSize, net_shape[0], net_shape[1], 3*len(list(mod_dirs.split(','))[:-1]))),
                             dtype=np.float32)
    gt_batch = np.zeros(((batchSize, n_class)), dtype=np.uint8)

    try:
        MIP_choices = MIP_choices.replace(' ', '')
        mip_list = MIP_choices.split(',')
    except:
        raise Exception('Unable to convert MIP_choices to a list of integers. Please check this argument.')

    while True:
        if shuff:
            shuffle(val_list)
        count = 0
        for scan_names in val_list:
            path_to_np = os.path.join(root_path,'np_files',scan_names[0].split(os.sep)[1]+'.npz')
            if scan_names[0].split(os.sep)[2].split('_')[0] == 'greece' or scan_names[0].split(os.sep)[2].split('_')[0] == 'irene':
                path_to_np = path_to_np[:-4] + '_' + scan_names[0].split(os.sep)[2].split('_')[0] + path_to_np[-4:]
            try:
                with np.load(path_to_np) as data:
                    imgs = np.stack((data['T1'], data['T2']), axis=-1) # TODO: Find modalities not hardcode
                    mask = data['mask']
            except:
                print('\nPre-made numpy array not found for {}.\n    Creating now...'.format(os.path.basename(path_to_np)))
                imgs, mask = convert_data_to_numpy(root_path=root_path, img_names=scan_names, mod_dirs=mod_dirs,
                                                       exp_name=exp_name, no_masks=False)
                if np.array_equal(imgs,np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')

            gt = int(scan_names[-1]) # GT depends on the experiment, do not save/load these, grab from train list.

            if imgs.shape[-2] < numSlices * (subSampAmt+1):
                imgs = np.pad(imgs, ((0,0), (0,0), (int(np.floor((numSlices * (subSampAmt+1) - imgs.shape[-2]) / 2)),
                                                    int(np.ceil((numSlices * (subSampAmt+1) - imgs.shape[-2]) / 2))),
                                     (0,0)), mode='symmetric')
                mask_largest = imgs.shape[-2] // 2
            else:
                _, _, mask_largest = find_mask_endpoints(mask)
                if mask_largest == -1:
                    mask_largest = imgs.shape[-2] // 2

                if mask_largest-numSlices//2 < 0 or mask_largest+numSlices//2+1 > imgs.shape[-2]:
                    mask_largest = imgs.shape[-2] // 2
                    if mask_largest - numSlices // 2 < 0 or mask_largest + numSlices // 2 + 1 > imgs.shape[-2]:
                        print('Unable to create validation example for {}.'.format(scan_names[0]))
                        continue

            if net.find('3d') != -1 or net.find('inflated') != -1:
                for i in range(imgs.shape[-1]):
                    img_batch[count, :,:,:, 3*i:3*i+3] = \
                        np.tile(np.expand_dims(
                            imgs[:, :, mask_largest-numSlices//2:mask_largest+numSlices//2+1, i], axis=-1), (1,1,1,3))
            else:
                cropped_imgs = imgs[:, :, mask_largest-numSlices//2:mask_largest+numSlices//2+1, :]
                assert len(mip_list) == cropped_imgs.shape[-1], \
                    'Different number of MIP_choices and imaging modalities given'
                for i, mip_choice in enumerate(mip_list):
                    if int(mip_choice) == 0:
                        img_batch[count, :, :, 3*i:3*i+3] = \
                            np.tile(np.expand_dims(np.min(cropped_imgs[:,:,:,i], axis=-1), axis=-1), (1,1,3))
                    elif int(mip_choice) == 1:
                        img_batch[count, :, :, 3*i:3*i+3] = \
                            np.tile(np.expand_dims(np.max(cropped_imgs[:,:,:,i], axis=-1), axis=-1), (1,1,3))
                    else:
                        raise Exception('Invalid choice for MIP_choices. Must be either 0 for min or 1 for max.')

            if n_class > 1:
                gt_batch[count, :] = get_one_hot(np.asarray(int(gt)), n_class)
            else:
                gt_batch[count, 0] = int(gt)

            count += 1
            if count % batchSize == 0:
                count = 0
                if img_batch.shape[3] == 1:
                    out_img_batch = np.squeeze(img_batch, axis=3)
                else:
                    out_img_batch = img_batch
                if net.find('caps') != -1:
                    if net.find('3d') != -1 or net.find('inflated') != -1:
                        out_recon_gt = out_img_batch[:,:,:,numSlices//2,:]
                    else:
                        out_recon_gt = out_img_batch
                    yield ([out_img_batch, gt_batch], [gt_batch, out_recon_gt])
                else:
                    yield (out_img_batch, gt_batch)

        if count != 0:
            if img_batch.shape[3] == 1:
                out_img_batch = np.squeeze(img_batch, axis=3)
            else:
                out_img_batch = img_batch
            if net.find('caps') != -1:
                if net.find('3d') != -1 or net.find('inflated') != -1:
                    out_recon_gt = out_img_batch[:,:,:,numSlices//2,:]
                else:
                    out_recon_gt = out_img_batch
                yield ([out_img_batch[:count, ...], gt_batch[:count, ...]],
                       [gt_batch[:count, ...], out_recon_gt[:count, ...]])
            else:
                yield (out_img_batch[:count, ...], gt_batch[:count, ...])

@threadsafe_generator
def generate_test_batches(root_path, test_list, net_shape, mod_dirs, exp_name, net, MIP_choices,
                                         n_class=1, batchSize=1, numSlices=1, subSampAmt=0, stride=1, downSampAmt=1):
    # Create placeholders for training
    if net.find('3d') != -1 or net.find('inflated') != -1:
        img_batch = np.zeros(((batchSize, net_shape[0], net_shape[1], numSlices, 3*len(list(mod_dirs.split(','))[:-1]))),
                             dtype=np.float32)
    else:
        img_batch = np.zeros(((batchSize, net_shape[0], net_shape[1], 3*len(list(mod_dirs.split(','))[:-1]))),
                             dtype=np.float32)

    try:
        MIP_choices = MIP_choices.replace(' ', '')
        mip_list = MIP_choices.split(',')
    except:
        raise Exception('Unable to convert MIP_choices to a list of integers. Please check this argument.')

    count = 0
    for scan_names in test_list:
        path_to_np = os.path.join(root_path,'np_files',scan_names[0].split(os.sep)[1]+'.npz')
        if scan_names[0].split(os.sep)[2].split('_')[0] == 'greece' or scan_names[0].split(os.sep)[2].split('_')[0] == 'irene':
            path_to_np = path_to_np[:-4] + '_' + scan_names[0].split(os.sep)[2].split('_')[0] + path_to_np[-4:]
        try:
            with np.load(path_to_np) as data:
                imgs = np.stack((data['T1'], data['T2']), axis=-1) # TODO: Find modalities not hardcode
                mask = data['mask']
        except:
            print('\nPre-made numpy array not found for {}.\n    Creating now...'.format(os.path.basename(path_to_np)))
            imgs, mask = convert_data_to_numpy(root_path=root_path, img_names=scan_names, mod_dirs=mod_dirs,
                                                   exp_name=exp_name, no_masks=False)
            if np.array_equal(imgs,np.zeros(1)):
                continue
            else:
                print('\nFinished making npz file.')

        if imgs.shape[-2] < numSlices * (subSampAmt+1):
            imgs = np.pad(imgs, ((0,0), (0,0), (int(np.floor((numSlices * (subSampAmt+1) - imgs.shape[-2]) / 2)),
                                                int(np.ceil((numSlices * (subSampAmt+1) - imgs.shape[-2]) / 2))),
                                 (0,0)), mode='symmetric')
            mask_largest = imgs.shape[-2] // 2
        else:
            _, _, mask_largest = find_mask_endpoints(mask)
            if mask_largest == -1:
                mask_largest = imgs.shape[-2] // 2

            if mask_largest-numSlices//2 < 0 or mask_largest+numSlices//2+1 > imgs.shape[-2]:
                mask_largest = imgs.shape[-2] // 2
                if mask_largest - numSlices // 2 < 0 or mask_largest + numSlices // 2 + 1 > imgs.shape[-2]:
                    raise Exception('Unable to create testing example for {}.\nThis must be corrected as it will throw off '
                                    'the indicies'.format(scan_names[0]))

        if net.find('3d') != -1 or net.find('inflated') != -1:
            for i in range(imgs.shape[-1]):
                img_batch[count, :,:,:, 3*i:3*i+3] = \
                    np.tile(np.expand_dims(
                            imgs[:, :, mask_largest-numSlices//2:mask_largest+numSlices//2+1, i], axis=-1), (1,1,1,3))
        else:
            cropped_imgs = imgs[:, :, mask_largest-numSlices//2:mask_largest+numSlices//2+1, :]
            assert len(mip_list) == cropped_imgs.shape[-1], \
                'Different number of MIP_choices and imaging modalities given'
            for i, mip_choice in enumerate(mip_list):
                if int(mip_choice) == 0:
                    img_batch[count, :, :, 3*i:3*i+3] = \
                        np.tile(np.expand_dims(np.min(cropped_imgs[:,:,:,i], axis=-1), axis=-1), (1,1,3))
                elif int(mip_choice) == 1:
                    img_batch[count, :, :, 3*i:3*i+3] = \
                        np.tile(np.expand_dims(np.max(cropped_imgs[:,:,:,i], axis=-1), axis=-1), (1,1,3))
                else:
                    raise Exception('Invalid choice for MIP_choices. Must be either 0 for min or 1 for max.')

        count += 1
        if count % batchSize == 0:
            count = 0
            if img_batch.shape[3] == 1:
                out_img_batch = np.squeeze(img_batch, axis=3)
            else:
                out_img_batch = img_batch
            yield (out_img_batch)

    if count != 0:
        if img_batch.shape[3] == 1:
            out_img_batch = np.squeeze(img_batch, axis=3)
        else:
            out_img_batch = img_batch
        yield (out_img_batch[:count,...])