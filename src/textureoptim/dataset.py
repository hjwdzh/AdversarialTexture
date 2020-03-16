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
from time import time
from tensorflow.python.client import device_lib

Dataset = collections.namedtuple("Dataset", "color_src,color_tar,uv_src,mask")
view_pairs = None
intrinsic = None
kernel = None

tex_dim_height = 1024
tex_dim_width = 1024

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


dictionary = {}
cached = False
def LoadDataByID(root, index):
    #if not index in dictionary:
    color_src_img = root + '/%05d_color.png'%(index)
    uv_src_img = root + '/%05d_uv.npz'%(index)
    depth_src_img = root + '/%05d_depth.npz'%(index)
    mask_src_img = root + '/%05d_mask.png'%(index)

    pose = root + '/%05d_pose.txt'%(index)
    color_src = cv2.imread(color_src_img) / 255.0
    uv_src = np.load(uv_src_img)['arr_0']
    depth_src = np.load(depth_src_img)['arr_0']
    mask_src = cv2.imread(mask_src_img,cv2.IMREAD_UNCHANGED) / 255.0
    world2cam = np.loadtxt(pose)
    return color_src.astype('float32'), uv_src.astype('float32'),\
        depth_src.astype('float32'), mask_src.astype('float32'),\
        world2cam.astype('float32')

def LoadChunk(filename):
    global cached, dictionary
    fn = filename
    if cached and (filename in dictionary):
        return dictionary[filename]
    root = filename[:-16]
    global view_pairs, intrinsic, kernel

    if not isinstance(filename, str):
        filename = filename.decode('utf-8')
    index = int(filename[-15:-10])
    root = filename[:-16]

    color_src, uv_src, depth_src, mask_src, world2cam_src\
        = LoadDataByID(root, index)

    rindex = random.choice(view_pairs[index])
    if rindex != index:
        color_tar, uv_tar, depth_tar, mask_tar, world2cam_tar\
            = LoadDataByID(root, rindex)
        
        cam2world_src = np.linalg.inv(world2cam_src)
        src2tar = np.transpose(np.dot(world2cam_tar, cam2world_src))

        y = np.linspace(0,IMAGE_HEIGHT-1,IMAGE_HEIGHT)
        x = np.linspace(0,IMAGE_WIDTH-1,IMAGE_WIDTH)
        xx, yy = np.meshgrid(x,y)

        fx = intrinsic[0]
        cx = intrinsic[2]
        fy = intrinsic[5]
        cy = intrinsic[6]

        x = (xx - cx) / fx * depth_src
        y = (yy - cy) / fy * depth_src
        coords = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,4))
        coords[:,:,0] = x
        coords[:,:,1] = y
        coords[:,:,2] = depth_src
        coords[:,:,3] = 1
        coords = np.dot(coords, src2tar)
        z_tar = coords[:,:,2]
        x = coords[:,:,0] / (1e-8+z_tar) * fx + cx
        y = coords[:,:,1] / (1e-8+z_tar) * fy + cy

        mask0 = (depth_src == 0)
        mask1 = (x < 0) + (y < 0) + (x >= IMAGE_WIDTH-1) + (y >= IMAGE_HEIGHT-1)
        lx = np.floor(x).astype('float32')
        ly = np.floor(y).astype('float32')
        rx = (lx + 1).astype('float32')
        ry = (ly + 1).astype('float32')
        sample_z1 = np.abs(z_tar\
            - cv2.remap(depth_tar, lx, ly, cv2.INTER_NEAREST))
        sample_z2 = np.abs(z_tar\
            - cv2.remap(depth_tar, lx, ry, cv2.INTER_NEAREST))
        sample_z3 = np.abs(z_tar\
            - cv2.remap(depth_tar, rx, ly, cv2.INTER_NEAREST))
        sample_z4 = np.abs(z_tar\
            - cv2.remap(depth_tar, rx, ry, cv2.INTER_NEAREST))
        mask2 = np.minimum(np.minimum(sample_z1, sample_z2),\
            np.minimum(sample_z3, sample_z4)) > 0.1

        mask_remap = (1 - (mask0 + mask1 + mask2 > 0)).astype('float32')

        map_x = x.astype('float32')
        map_y = y.astype('float32')

        color_tar_to_src = cv2.remap(color_tar, map_x, map_y, cv2.INTER_LINEAR)
        mask = (cv2.remap(mask_tar, map_x, map_y, cv2.INTER_LINEAR) > 0.99)\
            * mask_remap
        for j in range(3):
            color_tar_to_src[:,:,j] *= mask

    else:
        color_tar_to_src = color_src.copy()
        mask = mask_src.copy()

    color_src = (color_src * 2.0 - 1.0).astype('float32')
    color_tar_to_src = (color_tar_to_src * 2.0 - 1.0).astype('float32')
    uv_src[:,:,1] = 1 - uv_src[:,:,1]
    uv_src[:,:,0] *= tex_dim_width - 1
    uv_src[:,:,1] *= tex_dim_height - 1

    for i in range(3):
        color_src[:,:,i] *= mask
        color_tar_to_src[:,:,i] *= mask

    if cached:
        dictionary[fn] = (color_src, color_tar_to_src, uv_src,\
            np.reshape(mask,(mask.shape[0],mask.shape[1],1)))
        return dictionary[fn]

    return color_src, color_tar_to_src, uv_src,\
        np.reshape(mask,(mask.shape[0],mask.shape[1],1))

def create_dataset(parent_dir, texture_name, Cache=False):
    print(texture_name)
    p = cv2.imread(texture_name)
    
    global tex_dim_height, tex_dim_width, cached
    cached = Cache
    tex_dim_height = p.shape[0]
    tex_dim_width = p.shape[1]

    if parent_dir is None or not os.path.exists(parent_dir):
        raise Exception("input_dir does not exist")

    global view_pairs, intrinsic, kernel
    view_pairs = pickle.load(open(parent_dir + '/pose_pair.pkl','rb'))

    kernel = np.ones((11,11),np.uint8)
    color_paths = sorted(glob.glob(os.path.join(parent_dir, "*_color.png")))

    for i in range(len(view_pairs)):
        if type(view_pairs[i]) == type([]):
            p = view_pairs[i].copy()
        else:
            p = view_pairs[i].tolist()
        p.append(i)
        view_pairs[i] = np.array(p,dtype='int32')

    intrinsic = np.loadtxt(parent_dir + '/intrinsic.txt')
    intrinsic = np.reshape(intrinsic, [16])

    #color_paths = color_paths[:1]
    dataset = tf.data.Dataset.from_tensor_slices(color_paths)

    dataset = dataset.map(lambda filename: tf.py_func(LoadChunk, [filename],
        [tf.float32, tf.float32, tf.float32, tf.float32]))

    dataset = dataset.repeat()
    dataset = dataset.batch(1)
    dataset = dataset.shuffle(1)
    dataset = dataset.prefetch(1)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    color_src = next_element[0]
    color_tar = next_element[1]
    uv_src = next_element[2]
    mask = next_element[3]

    color_src.set_shape([1,IMAGE_HEIGHT,IMAGE_WIDTH,3])
    color_tar.set_shape([1,IMAGE_HEIGHT,IMAGE_WIDTH,3])
    uv_src.set_shape([1,IMAGE_HEIGHT,IMAGE_WIDTH,2])
    mask.set_shape([1,IMAGE_HEIGHT,IMAGE_WIDTH,1])

    return Dataset(color_src=color_src,\
        color_tar=color_tar, uv_src=uv_src, mask=mask)
