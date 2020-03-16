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

from dataset import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--output_dir", help="path to folder containing images")
parser.add_argument("--gan_weight", default=1.0, type=float,help="path to folder containing images")
parser.add_argument("--l1_weight", default=10.0, type=float,help="path to folder containing images")
parser.add_argument("--lr_G", type=float, default=1e-3, help="initial learning rate for adam")
parser.add_argument("--lr_D", type=float, default=1e-4, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--initialized", type=int, default=1, help="")

a = parser.parse_args()

filename = a.input_dir.split('/')[-1]
initial_file = a.input_dir + '/texture.png'
dataset = create_dataset(a.input_dir, initial_file, Cache=True)

if not os.path.exists(a.output_dir):
    os.mkdir(a.output_dir)

if a.initialized == 1:
    model = create_model(dataset,a.lr_D,a.lr_G,a.beta1,initial_file)
else:
    model = create_model(dataset,a.lr_D,a.lr_G,a.beta1,None)

logdir = a.output_dir
max_steps = 4001

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(max_steps):
    fetches = {
        "train": model.train,
        "global_step": model.global_step,
        'lossG': model.gen_loss_GAN,
        'lossL1': model.gen_loss_L1,
        'lossD': model.discrim_loss,
    }
    if step % 100 == 0:
        fetches['outputs'] = model.texture

    results = sess.run(fetches)

    if step % 100 == 0:
        cv2.imwrite('%s/%06d.png'%(a.output_dir,step//100), (np.clip(results['outputs'][0] * 0.5 + 0.5,0,1) * 255).astype('uint8'))

    if step % 10 == 0:
        print('iter=%d, lossL1=%.4f lossG=%.4f lossD=%.4f'%(results["global_step"], results['lossL1'], results['lossG'], results['lossD']))

    if step == max_steps - 1:
        cv2.imwrite('%s/%s.png'%(a.output_dir,filename), (np.clip(results['outputs'][0] * 0.5 + 0.5,0,1) * 255).astype('uint8'))
