
import os
import loader
import rasterizer
import painter
import sys
import numpy as np
import skimage.io as sio
import cv2
from ctypes import *
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import pickle
sys.path.insert(1, '..')
from config import *

Render = cdll.LoadLibrary('./CudaRender/libRender.so')
def SetMesh(V, F):
  handle = Render.SetMesh(c_void_p(V.ctypes.data), c_void_p(F.ctypes.data), V.shape[0], F.shape[0])
  return handle

def render(handle, world2cam):
  Render.SetTransform(handle, c_void_p(world2cam.ctypes.data))
  Render.Render(handle);

def getDepth():
  depth = np.zeros((480,640), dtype='float32')
  Render.GetDepth(c_void_p(depth.ctypes.data))

  return depth

def getVMap(handle):
  vindices = np.zeros((480, 640, 3), dtype='int32')
  vweights = np.zeros((480, 640, 3), dtype='float32')
  findices = np.zeros((480, 640), dtype='int32')

  Render.GetVMap(handle, c_void_p(vindices.ctypes.data), c_void_p(vweights.ctypes.data), c_void_p(findices.ctypes.data))

  return vindices, vweights, findices

def Project(info, world2cam, point):
	points = np.transpose(np.dot(world2cam[0:3,0:3], np.transpose(point)))
	points[:,0] += world2cam[0,3]
	points[:,1] += world2cam[1,3]
	points[:,2] += world2cam[2,3]
	x = points[:,0] / points[:,2] * info[0,0] + info[0,2]
	y = points[:,1] / points[:,2] * info[1,1] + info[1,2]
	return np.array([x,y,points[:,2]])

tex_dim = 1024

sens_file = data_path + '/scan/' + sys.argv[1] + '_video.sens'
obj_file = data_path + '/shape/' + sys.argv[1] + '.obj'

V, F, VT, FT, VN, FN = loader.LoadOBJ(obj_file)

colors, depths, cam2worlds, intrinsic = loader.LoadSens(sens_file)

if not os.path.exists(data_path + '/ObjectScan_video'):
	os.mkdir(data_path + '/ObjectScan_video')
if not os.path.exists(data_path + '/ObjectScan_video/%s'%(sys.argv[1])):
	os.mkdir(data_path + '/ObjectScan_video/%s'%(sys.argv[1]))

fp = open(data_path + '/ObjectScan_video/%s/pose_pair.pkl'%(sys.argv[1]),'wb')
t = []
min_len = 10000
for i in range(cam2worlds.shape[0]):
	a = []
	for j in range(cam2worlds.shape[0]):
		angle = np.dot(cam2worlds[i,:,2], cam2worlds[j,:,2])
		if angle > np.cos(15.0 / 180.0 * np.pi):
			a.append(j)
	if len(a) < min_len:
		min_len = len(a)
	t.append(a)

pickle.dump(t, fp)
fp.close()

Render.InitializeCamera(640,480,
	c_float(intrinsic[0,0]), c_float(intrinsic[1,1]), c_float(intrinsic[0,2]), c_float(intrinsic[1,2]))

vweights, findices = rasterizer.RasterizeTexture(VT, FT, tex_dim, tex_dim)

print('generate textiles')
points, normals, coords = rasterizer.GeneratePoints(V, F, VN, FN, vweights, findices)

print('paint colors')
point_colors = np.zeros((points.shape[0], 4), dtype='float32')

for i in range(colors.shape[0]):
	painter.ProjectPaint(points, normals, point_colors, colors[i], depths[i], np.linalg.inv(cam2worlds[i]).astype('float32'), intrinsic)

print('prepare final texture')
original_texture = np.zeros((tex_dim, tex_dim, 3), dtype='uint8')
painter.PaintToTexturemap(original_texture, point_colors, coords)

sio.imsave(data_path + '/ObjectScan_video/%s/texture.png'%(sys.argv[1]), original_texture)
np.savetxt(data_path + '/ObjectScan_video/%s/intrinsic.txt'%(sys.argv[1]), intrinsic)

context = SetMesh(V, F)

for i in range(colors.shape[0]):
	print('view %d of %d...'%(i, colors.shape[0]))
	world2cam = np.linalg.inv(cam2worlds[i]).astype('float32')
	render(context, world2cam)
	depth = getDepth()

	vindices, vweights, findices = getVMap(context)

	uv = np.zeros((findices.shape[0], findices.shape[1], 3), dtype='float32')
	for j in range(2):
		uv[:,:,j] = 0
		for k in range(3):
			vind = FT[findices][:,:,k]
			uv_ind = VT[vind][:,:,j]
			uv[:,:,j] += vweights[:,:,k] * uv_ind

	#mask0 = ((np.abs(depth - depths[i]*1e-3) < 0.05) * (depths[i] > 0)).astype('uint8')

	mask = (findices != -1)# * mask0

	style = cv2.resize(colors[i], (depths[i].shape[1], depths[i].shape[0]), interpolation=cv2.INTER_LINEAR)
	for j in range(3):
		style[:,:,j] *= mask
		uv[:,:,j] *= mask

	depth *= mask

	sio.imsave(data_path + '/ObjectScan_video/%s/%05d_mask.png'%(sys.argv[1], i), (mask*255).astype('uint8'))
	sio.imsave(data_path + '/ObjectScan_video/%s/%05d_color.png'%(sys.argv[1], i), style)
	np.savez_compressed(data_path + '/ObjectScan_video/%s/%05d_depth.npz'%(sys.argv[1], i), depth)
	np.savez_compressed(data_path + '/ObjectScan_video/%s/%05d_uv.npz'%(sys.argv[1], i), uv[:,:,:2])
	np.savetxt(data_path + '/ObjectScan_video/%s/%05d_pose.txt'%(sys.argv[1], i), world2cam)
