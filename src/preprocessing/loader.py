from ctypes import *
import numpy as np
Sens = cdll.LoadLibrary('./Rasterizer/libSens.so')

def LoadSens(filename):
	print(filename)
	Sens.Parse(c_char_p(filename.encode('utf8')))
	depth_width = Sens.DW()
	depth_height = Sens.DH()
	color_width = Sens.CW()
	color_height = Sens.CH()
	frames = Sens.Frames()
	depths = np.zeros((frames, depth_height, depth_width), dtype='float32')
	colors = np.zeros((frames, color_height, color_width, 3), dtype='uint8')
	cam2worlds = np.zeros((frames, 4, 4), dtype='float32')
	intrinsic = np.zeros((4,4), dtype='float32')

	Sens.GetData(c_void_p(depths.ctypes.data), c_void_p(colors.ctypes.data),\
		c_void_p(cam2worlds.ctypes.data), c_void_p(intrinsic.ctypes.data))
	Sens.Clear()
	depths = np.nan_to_num(depths)
	return colors, depths, cam2worlds, intrinsic

def LoadOBJ(filename):
	lines = [l.strip() for l in open(filename)]
	V = []
	VT = []
	VN = []
	F = []
	FT = []
	FN = []
	for l in lines:
		words = [w for w in l.split(' ') if w != '']
		if words[0] == 'v':
			V.append([float(words[1]), float(words[2]), float(words[3])])
		elif words[0] == 'vt':
			VT.append([float(words[1]), float(words[2])])
		elif words[0] == 'vn':
			VN.append([float(words[1]), float(words[2]), float(words[3])])
		elif words[0] == 'f':
			f = []
			ft = []
			fn = []
			for j in range(1, 4):
				w = words[j].split('/')
				f.append(int(w[0])-1)
				ft.append(int(w[1])-1)
				fn.append(int(w[2])-1)
			F.append(f)
			FT.append(ft)
			FN.append(fn)

	V = np.array(V, dtype='float32')
	VT = np.array(VT, dtype='float32')
	VN = np.array(VN, dtype='float32')
	F = np.array(F, dtype='int32')
	FT = np.array(FT, dtype='int32')
	FN = np.array(FN, dtype='int32')

	return V, F, VT, FT, VN, FN

if __name__ == "__main__":
	import sys
	LoadSens(sys.argv[1])
	V,F,VT,FT,_,_ = LoadOBJ(sys.argv[2])