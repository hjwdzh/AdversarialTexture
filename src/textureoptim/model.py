from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import argparse
import glob
import cv2
import numpy as np
import pickle
import random
import collections
import time
from tensorflow.python.client import device_lib

EPS = 1e-12
Model = collections.namedtuple("Model", "predict_real,predict_fake,\
    predict_mask,discrim_loss,discrim_loss_final,gen_loss_GAN,gen_loss_L1,\
    outputs,texture,train,l1_weight,global_step")

def create_model(data,lr_D,lr_G,beta1,initial_file):

    def lrelu(x, a):
        with tf.name_scope("lrelu"):
            x = tf.identity(x)
            return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

    def discrim_conv(batch_input, out_channels, stride):
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]],\
            mode="CONSTANT")
        return tf.layers.conv2d(padded_input, out_channels, kernel_size=4,\
            strides=(stride, stride), padding="valid",\
            kernel_initializer=tf.random_normal_initializer(0, 0.02))

    def discrim_conv_mask(batch_input, stride):
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]],\
            mode="CONSTANT")
        return tf.layers.conv2d(padded_input, 1, kernel_size=4,\
            strides=(stride, stride), padding="valid",\
            kernel_initializer=tf.constant_initializer(1.0/16))

    def create_texture(initial_file):
        if not initial_file is None:
            t = cv2.imread(initial_file)
            p = np.sum(t, axis=2) > 0
            t = (t / 255.0 * 2.0 - 1.0)
            for j in range(3):
                t[:,:,j] *= p
            t = np.reshape(t, (1, t.shape[0], t.shape[1], 3))

        else:
            t = np.zeros((1,1024,1024,3))

        texture = tf.get_variable("texture", dtype=tf.float32,
          initializer=t.astype('float32'))
        return texture


    def create_discriminator(input, mask):
        n_layers = 3
        ndf = 64
        layers = []
        layers_mask = []

        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, ndf, stride=2)
            convolved_mask = discrim_conv_mask(mask, stride=2)

            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)
            layers_mask.append(convolved_mask)

        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = ndf * min(2**(i+1), 2)
                stride = 1 if i == n_layers - 1 else 2
                convolved = discrim_conv(layers[-1],\
                    out_channels, stride=stride)
                convolved_mask = discrim_conv_mask(layers_mask[-1],\
                    stride=stride)
                rectified = lrelu(convolved, 0.2)
                layers.append(rectified)
                layers_mask.append(convolved_mask)

        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            convolved_mask = discrim_conv_mask(convolved_mask, stride=stride)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1], tf.dtypes.cast(convolved_mask > 0.1, tf.float32)

    global_step = tf.train.get_or_create_global_step()

    l1_weight = float(10.0) * (float(0.8) **\
        tf.dtypes.cast((global_step//960),tf.float32))
    texture = create_texture(initial_file)
    outputs = tf.contrib.resampler.resampler(\
        texture, data.uv_src, name='resampler')
    mask3 = tf.concat((data.mask, data.mask, data.mask), axis=3)
    outputs = outputs * mask3

    offsets = tf.random.uniform([2],0,70, dtype=tf.int32)
    sy = offsets[0]
    sx = offsets[1]
    # create two copies of discriminator,
    # one for real pairs and one for fake pairs
    # they share the same underlying variables
    input = tf.concat([data.color_src, data.color_tar - data.color_src], axis=3)
    #input, ops_real = concat_image_pool(input)
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real, mask_real = create_discriminator(input[:,sy:,sx:,:],\
                data.mask[:,sy:,sx:,:])


    input = tf.concat([data.color_src, outputs - data.color_src], axis=3)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake, mask_fake = create_discriminator(input[:,sy:,sx:,:],\
                data.mask[:,sy:,sx:,:])

    style_weight = 1e-6
    with tf.name_scope("generator_loss"):
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_sum(tf.reduce_sum(\
            tf.abs(data.color_tar[:,sy:,sx:,:] - outputs[:,sy:,sx:,:]),axis=3)\
            * data.mask[:,sy:,sx:,0]) / (tf.reduce_sum(data.mask[:,sy:,sx:])\
            * float(3.0) + EPS)
        gen_loss = gen_loss_L1 * l1_weight + gen_loss_GAN

    with tf.name_scope("discriminator_loss"):
        discrim_loss = tf.reduce_sum((-(tf.log(predict_real + EPS)\
            + tf.log(1 - predict_fake + EPS))) * mask_real)\
            / (tf.reduce_sum(mask_real) + EPS)

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables()\
            if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(lr_D, beta1)
        discrim_loss_final = tf.cond(discrim_loss > gen_loss_GAN,\
            lambda: discrim_loss, lambda: discrim_loss * 0)
        discrim_grads_and_vars = discrim_optim.compute_gradients(\
            discrim_loss_final, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables()\
                if var.name.startswith("texture")]
            gen_optim = tf.train.AdamOptimizer(lr_G, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(\
                gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        predict_mask=mask_fake,
        discrim_loss=discrim_loss,
        discrim_loss_final=discrim_loss_final,
        gen_loss_GAN=gen_loss_GAN,
        gen_loss_L1=gen_loss_L1,
        outputs=outputs,
        texture=texture,
        train=tf.group(gen_train, discrim_train, incr_global_step),
        l1_weight=l1_weight,
        global_step=global_step,
    )
