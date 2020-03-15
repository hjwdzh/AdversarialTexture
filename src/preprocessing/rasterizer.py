from ctypes import *
import numpy as np
import os
libpath = os.path.dirname(os.path.abspath(__file__))

Rasterizer = cdll.LoadLibrary(libpath + '/Rasterizer/libRasterizer.so')

def RasterizeTexture(VT, FT, height, width):
	color = np.zeros((height,width,3), dtype='float32')
	zbuffer = np.zeros((height, width), dtype='float32')
	findices = np.zeros((height, width), dtype='int32')
	findices[:,:] = -1
	Rasterizer.RasterizeTexture(c_void_p(VT.ctypes.data), c_void_p(FT.ctypes.data), c_void_p(color.ctypes.data),
		c_void_p(zbuffer.ctypes.data), c_void_p(findices.ctypes.data), c_int(FT.shape[0]), width, height)

	return color, findices


def RasterizeImage(V, F, width, height, intrinsic, world2cam):
	vweights = np.zeros((height, width, 3), dtype='float32')
	findices = np.zeros((height, width), dtype='int32')
	zbuffer = np.zeros((height, width), dtype='float32')
	findices[:,:] = -1
	Rasterizer.RasterizeImage(c_void_p(V.ctypes.data), c_void_p(F.ctypes.data), c_void_p(world2cam.ctypes.data), c_void_p(intrinsic.ctypes.data),
		c_void_p(vweights.ctypes.data), c_void_p(zbuffer.ctypes.data), c_void_p(findices.ctypes.data), c_int(F.shape[0]), c_int(width), c_int(height))
	return vweights, findices

def GeneratePoints(V, F, VN, FN, vweights, findices):
	num_points = np.sum(findices >= 0)
	points = np.zeros((num_points, 3), dtype='float32')
	normals = np.zeros((num_points, 3), dtype='float32')
	coords = np.zeros((num_points, 2), dtype='int32')
	Rasterizer.GenerateTextiles(c_void_p(V.ctypes.data), c_void_p(F.ctypes.data), c_void_p(VN.ctypes.data), c_void_p(FN.ctypes.data),
		c_void_p(points.ctypes.data), c_void_p(normals.ctypes.data), c_void_p(coords.ctypes.data),
		c_void_p(findices.ctypes.data), c_void_p(vweights.ctypes.data), c_int(findices.shape[1]), c_int(findices.shape[0]))
	return points, normals, coords

def RenderUV(vweights, findices, VT, FT):
	uv = np.zeros((vweights.shape[0], vweights.shape[1], 3), dtype='float32')
	#void RenderUV(glm::vec3* uv, glm::vec3* vweights, int* findices, glm::vec2* VT, glm::ivec3* FT, int width, int height)

	Rasterizer.RenderUV(c_void_p(uv.ctypes.data), c_void_p(vweights.ctypes.data), c_void_p(findices.ctypes.data), 
		c_void_p(VT.ctypes.data), c_void_p(FT.ctypes.data),
		c_int(uv.shape[1]), c_int(uv.shape[0]))
	return uv