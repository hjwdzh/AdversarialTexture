from ctypes import *
import numpy as np
Painter = cdll.LoadLibrary('./Rasterizer/libPainter.so')

def ProjectPaint(points, normals, point_colors, color, depth, world2cam, intrinsic):
	Painter.ProjectPaint(c_void_p(points.ctypes.data), c_void_p(normals.ctypes.data), c_void_p(point_colors.ctypes.data),
		c_void_p(color.ctypes.data), c_void_p(depth.ctypes.data), c_void_p(world2cam.ctypes.data), c_void_p(intrinsic.ctypes.data),
		c_int(points.shape[0]), c_int(depth.shape[1]), c_void_p(depth.shape[0]), c_void_p(color.shape[1]), c_void_p(color.shape[0]))

def PaintToTexturemap(texturemap, point_colors, coords):
	Painter.PaintToTexturemap(c_void_p(texturemap.ctypes.data), c_void_p(point_colors.ctypes.data), c_void_p(coords.ctypes.data),
		c_int(point_colors.shape[0]), c_int(texturemap.shape[1]), c_int(texturemap.shape[0]))

def PaintToViewNorm(points_cam, normals_cam, mask, depth, coords, textureToImage):
	Painter.PaintToViewNorm(c_void_p(textureToImage.ctypes.data), c_void_p(points_cam.ctypes.data), c_void_p(normals_cam.ctypes.data), c_void_p(mask.ctypes.data),
		c_void_p(depth.ctypes.data), c_void_p(coords.ctypes.data), c_int(points_cam.shape[0]), c_int(mask.shape[0]), c_int(mask.shape[1]), c_int(textureToImage.shape[1]))

def PaintToView(points_cam, mask, depth, coords, textureToImage):
	Painter.PaintToView(c_void_p(textureToImage.ctypes.data), c_void_p(points_cam.ctypes.data), c_void_p(mask.ctypes.data),
		c_void_p(depth.ctypes.data), c_void_p(coords.ctypes.data), c_int(points_cam.shape[0]), c_int(mask.shape[0]), c_int(mask.shape[1]), c_int(textureToImage.shape[1]))
