import numpy as np
import sys
import skimage.io as sio
import os
#from gen_poses import GetPoses,WavePose
import shutil
from objloader import LoadTextureOBJ
import render
import objloader


input_obj = sys.argv[1]
V, F, VT, FT, VN, FN, face_mat, kdmap = objloader.LoadTextureOBJ(input_obj)

# set up camera information
info = {'Height':960, 'Width':1280, 'fx':575*2, 'fy':575*2, 'cx':640, 'cy':480}
render.setup(info)

context = render.SetMesh(V, F)

cam2world = np.array([[ 0.85408425,  0.31617427, -0.375678  ,  0.56351697 * 2],
       [ 0.        , -0.72227067, -0.60786998,  0.91180497 * 2],
       [-0.52013469,  0.51917219, -0.61688   ,  0.92532003 * 2],
       [ 0.        ,  0.        ,  0.        ,  1.        ]], dtype=np.float32)

world2cam = np.linalg.inv(cam2world).astype('float32')
render.render(context, world2cam)
depth = render.getDepth(info)
vindices, vweights, findices = render.getVMap(context, info)

sio.imsave('depth.png', depth / np.max(depth))
sio.imsave('vweights.png', vweights)