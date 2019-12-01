'''
INN: Inflated Neural Networks for IPMN Diagnosis
Original Paper by Rodney LaLonde, Irene Tanner, Katerina Nikiforaki, Georgios Z. Papadakis, Pujan Kandel,
Candice W. Bolan, Michael B. Wallace, Ulas Bagci
(https://link.springer.com/chapter/10.1007/978-3-030-32254-0_12, https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for testing models. Please see the README for details about testing.
'''

from __future__ import print_function, division

import os
import csv

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from keras import backend as K
K.set_image_data_format('channels_last')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from load_3D_data import generate_test_batches

def test(args, test_list, eval_model, net_input_shape):
    # Set the path to the testing weights for the model, either based on user or on training
    if args.test_weights_path != '':
        output_dir = os.path.join(args.data_root_dir, 'results', args.exp_name, args.net,
                                  'split_{}'.format(args.split_num), os.path.basename(args.test_weights_path)[:-5])
        try:
            eval_model.load_weights(args.test_weights_path)
        except Exception as e:
            print(e)
            raise Exception('Failed to load weights from training.')
    else:
        output_dir = os.path.join(args.data_root_dir, 'results', args.exp_name, args.net,
                                  'split_{}'.format(args.split_num), args.output_name + '_model_' + args.time)
        try:
            eval_model.load_weights(os.path.join(args.check_dir, args.output_name + '_model_' + args.time + '.hdf5'))
        except Exception as e:
            print(e)
            raise Exception('Failed to load weights from training.')

    # Create an output directory for saving results
    try:
        os.makedirs(output_dir)
    except:
        pass

    # Create a CSV for saving the results
    with open(os.path.join(output_dir, args.save_prefix + 'scores.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        row = ['Scan Name', 'Prediction', 'Label']
        writer.writerow(row)

        # Testing the network
        print('Testing... This will take some time...')
        output_array = []
        for test_sample in test_list:
            output_array.append(eval_model.predict_generator(
                generate_test_batches(root_path=args.data_root_dir, test_list=[test_sample],
                                      net_shape=net_input_shape, mod_dirs=args.modality_dir_list,
                                      exp_name=args.exp_name, net=args.net, MIP_choices=args.MIP_choices,
                                      batchSize=1, numSlices=args.slices, subSampAmt=0, stride=1),
                steps=1, max_queue_size=1, workers=1, use_multiprocessing=False, verbose=1))

        # Convert the network output to predictions
        if args.num_classes > 2:
            output = np.argmax(np.squeeze(np.asarray(output_array, dtype=np.float32)), axis=1)
        else:
            output = np.copy(np.asarray(output_array, dtype=np.float32)[:,0])
            output[output < 0.5] = 0
            output[output >= 0.5] = 1

        # Save the names, predictions, and GTs to a file
        name_list = [x.split('/')[1] for x in np.asarray(test_list)[:,0]]
        gt_list = np.asarray(test_list)[:,-1].astype(int)
        assert len(gt_list) == len(output), 'Different number of outputs and ground truth labels in testing.'

        print(zip(name_list, np.squeeze(np.asarray(output_array)), gt_list))
        print('Accuracy: {}'.format(accuracy_score(gt_list, output)))
        print('F1 Score: {}'.format(f1_score(gt_list, output, average='macro')))
        print('Precision: {}'.format(precision_score(gt_list, output, average='macro')))
        print('Recall: {}'.format(recall_score(gt_list, output, average='macro')))
        print('Confusion matrix:')
        print(confusion_matrix(gt_list, output))
        print('Classification Report:')
        print(classification_report(gt_list, output))

        writer.writerows(zip(name_list, np.squeeze(np.asarray(output_array)), gt_list))
        writer.writerow(['Accuracy:','{}'.format(accuracy_score(gt_list, output))])
        writer.writerow(['F1 Score:','{}'.format(f1_score(gt_list, output, average='macro'))])
        writer.writerow(['Precision:','{}'.format(precision_score(gt_list, output, average='macro'))])
        writer.writerow(['Recall:','{}'.format(recall_score(gt_list, output, average='macro'))])
        writer.writerow(['Confusion matrix:'])
        writer.writerow(confusion_matrix(gt_list, output))
        writer.writerow(['Classification Report:'])
        writer.writerow(classification_report(gt_list, output))

    print('Done.')
