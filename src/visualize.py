import os
import cv2
import numpy as np

from config import *

shapes = os.listdir(data_path + '/result/')

if not os.path.exists(data_path + '/visual/'):
	os.mkdir(data_path + '/visual')

views = {'chair00':0, 'chair01':60, 'chair02':0, 'chair03':0, 'chair04':120,
	'chair05':300, 'chair06':50, 'chair07':240, 'chair08':300, 'chair09':0,
	'chair10':180, 'chair11':180, 'chair12':0, 'chair13':0, 'chair14':0,
	'chair15':0, 'chair16':0, 'chair17':0, 'chair18':0, 'chair19':0,
	'chair20':0, 'chair21':0, 'chair22':0, 'chair23':0, 'chair24':0,
	'chair25':0, 'chair26':0, 'chair27':180, 'chair28':0, 'chair29':60,
	'chair30':0, 'chair31':0, 'chair32':0, 'chair33':0, 'chair34':0}


for s in shapes:
	our_texture = cv2.imread(data_path + '/result/' + s + '/000040.png')
	l2_texture = cv2.imread(data_path + '/shape/' + s + '.png')

	if not s in views:
		continue
	view = views[s]
	if not os.path.exists(data_path + '/ObjectScan_video/'\
		+ s + '/%05d_uv.npz'%(view)):
		continue
	uv = np.load(data_path + '/ObjectScan_video/' + s\
		+ '/%05d_uv.npz'%(view))['arr_0']

	mask = np.sum(np.abs(uv), axis=2) > 0

	l2_img = cv2.remap(l2_texture, uv[:,:,0] * (l2_texture.shape[1] - 1),\
		(1 - uv[:,:,1]) * (l2_texture.shape[0] - 1), cv2.INTER_LINEAR)
	our_img = cv2.remap(our_texture, uv[:,:,0] * (our_texture.shape[1] - 1),\
		(1 - uv[:,:,1]) * (our_texture.shape[0] - 1), cv2.INTER_LINEAR)
	
	for i in range(3):
		l2_img[:,:,i] = l2_img[:,:,i]*mask+(255 * (1 - mask)).astype('uint8')
		our_img[:,:,i] = our_img[:,:,i]*mask+(255 * (1 - mask)).astype('uint8')

	output_path = data_path + '/visual/' + s + '_l2.png'
	cv2.imwrite(output_path, l2_img)

	output_path = data_path + '/visual/' + s + '_our.png'
	cv2.imwrite(output_path, our_img)
