# -*- coding: utf-8 -*-
'''
INN: Inflated Neural Networks for IPMN Diagnosis
Original Paper by Rodney LaLonde, Irene Tanner, Katerina Nikiforaki, Georgios Z. Papadakis, Pujan Kandel,
Candice W. Bolan, Michael B. Wallace, Ulas Bagci
(https://link.springer.com/chapter/10.1007/978-3-030-32254-0_12, https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the definitions of custom loss functions not present in the default Keras.
'''
from __future__ import print_function

import tensorflow as tf


def binary_crossentropy_loss():
    def binary_crossentropy(target, output, from_logits=False):
        """Binary crossentropy between an output tensor and a target tensor.
        # Arguments
            target: A tensor with the same shape as `output`.
            output: A tensor.
            from_logits: Whether `output` is expected to be a logits tensor.
                By default, we consider that `output`
                encodes a probability distribution.
        # Returns
            A tensor.
        """
        # Note: tf.nn.sigmoid_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # transform back to logits
            _epsilon = tf.convert_to_tensor(value=1e-7, dtype=output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
            output = tf.math.log(output / (1 - output))

        return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

    return binary_crossentropy


def weighted_binary_crossentropy_loss(pos_weight):
    # pos_weight: A coefficient to use on the positive examples.
    def weighted_binary_crossentropy(target, output, from_logits=False):
        """Binary crossentropy between an output tensor and a target tensor.
        # Arguments
            target: A tensor with the same shape as `output`.
            output: A tensor.
            from_logits: Whether `output` is expected to be a logits tensor.
                By default, we consider that `output`
                encodes a probability distribution.
        # Returns
            A tensor.
        """
        # Note: tf.nn.sigmoid_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # transform back to logits
            _epsilon = tf.convert_to_tensor(value=1e-7, dtype=output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
            output = tf.math.log(output / (1 - output))

        return tf.nn.weighted_cross_entropy_with_logits(labels=target, logits=output, pos_weight=pos_weight)
    return weighted_binary_crossentropy


def categorical_crossentropy_loss():
    def categorical_crossentropy(target, output, from_logits=False, axis=-1):
        """Categorical crossentropy between an output tensor and a target tensor.
        # Arguments
            target: A tensor of the same shape as `output`.
            output: A tensor resulting from a softmax
                (unless `from_logits` is True, in which
                case `output` is expected to be the logits).
            from_logits: Boolean, whether `output` is the
                result of a softmax, or is a tensor of logits.
            axis: Int specifying the channels axis. `axis=-1`
                corresponds to data format `channels_last`,
                and `axis=1` corresponds to data format
                `channels_first`.
        # Returns
            Output tensor.
        # Raises
            ValueError: if `axis` is neither -1 nor one of
                the axes of `output`.
        """
        output_dimensions = list(range(len(output.get_shape())))
        if axis != -1 and axis not in output_dimensions:
            raise ValueError(
                '{}{}{}'.format(
                    'Unexpected channels axis {}. '.format(axis),
                    'Expected to be -1 or one of the axes of `output`, ',
                    'which has {} dimensions.'.format(len(output.get_shape()))))
        # Note: tf.nn.softmax_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # scale preds so that the class probas of each sample sum to 1
            output /= tf.reduce_sum(input_tensor=output, axis=axis, keepdims=True)
            # manual computation of crossentropy
            _epsilon = tf.convert_to_tensor(value=1e-7, dtype=output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
            return - tf.reduce_sum(input_tensor=target * tf.math.log(output), axis=axis)
        else:
            return tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(target), logits=output)

    return categorical_crossentropy
