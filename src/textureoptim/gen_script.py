import os
import sys
sys.path.insert(1, '..')
from config import *

input_folder = data_path + '/ObjectScan_video'
output_folder = data_path + '/result'

files = os.listdir(input_folder)
fp = open('run_all.sh','w')
for f in files:
	fp.write('python3 optim.py --input_dir %s/%s --output_dir %s/%s\n'%\
		(input_folder, f, output_folder, f))

fp.close()
