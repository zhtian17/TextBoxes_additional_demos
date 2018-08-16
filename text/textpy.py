import numpy as np
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
# %matplotlib inline
import time
import math
from nms import nms
from crop_image import crop_image

# Make sure that caffe is on the python path:
caffe_root = './'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
caffe.set_device(3)

import subprocess

import glob
import cv2
import skimage.io

def get_config():
	config = {
		'model_def' : '../../models/deploy.prototxt',
		'model_weights' : '../../models/model_icdar15.caffemodel',
		#'img_dir' : '../../demo_images/test/',
		'det_visu_dir' : '../../demo_images/std_result_img/',
		'det_save_dir' : '../../demo_images/std_result_txt/',
		'crop_dir' : '../../demo_images/crops/',
		'lexicon_path' : '../../crnn/data/icdar_generic_lexicon.txt',
		'use_lexcion' : True,
		#'input_height' : 500,
		#'input_width' : 700,
		'overlap_threshold' : 0.2,
		'det_score_threshold' : 0.5,
		'f_score_threshold' : 0.7,
		'visu_detection' : True
		}
	return config

def prepare_network(config):
	net = caffe.Net(config['model_def'],	 # defines the structure of the model
                config['model_weights'],  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
	return net


def extract_detections(detections, det_score_threshold, image_height, image_width):
	det_conf = detections[0,0,:,2]
	det_x1 = detections[0,0,:,7]
	det_y1 = detections[0,0,:,8]
	det_x2 = detections[0,0,:,9]
	det_y2 = detections[0,0,:,10]
	det_x3 = detections[0,0,:,11]
	det_y3 = detections[0,0,:,12]
	det_x4 = detections[0,0,:,13]
	det_y4 = detections[0,0,:,14]
	# Get detections with confidence higher than 0.6.
	top_indices = [i for i, conf in enumerate(det_conf) if conf >= det_score_threshold]
	top_conf = det_conf[top_indices]
	top_x1 = det_x1[top_indices]
	top_y1 = det_y1[top_indices]
	top_x2 = det_x2[top_indices]
	top_y2 = det_y2[top_indices]
	top_x3 = det_x3[top_indices]
	top_y3 = det_y3[top_indices]
	top_x4 = det_x4[top_indices]
	top_y4 = det_y4[top_indices]

	bboxes=[]
	for i in xrange(top_conf.shape[0]):
		x1 = int(round(top_x1[i] * image_width))
		y1 = int(round(top_y1[i] * image_height))
		x2 = int(round(top_x2[i] * image_width))
		y2 = int(round(top_y2[i] * image_height))
		x3 = int(round(top_x3[i] * image_width))
		y3 = int(round(top_y3[i] * image_height))
		x4 = int(round(top_x4[i] * image_width))
		y4 = int(round(top_y4[i] * image_height))
		x1 = max(1, min(x1, image_width - 1))
		x2 = max(1, min(x2, image_width - 1))
		x3 = max(1, min(x3, image_width - 1))
		x4 = max(1, min(x4, image_width - 1))
		y1 = max(1, min(y1, image_height - 1))
		y2 = max(1, min(y2, image_height - 1))
		y3 = max(1, min(y3, image_height - 1))
		y4 = max(1, min(y4, image_height - 1))
		score = top_conf[i]
		bbox=[x1,y1,x2,y2,x3,y3,x4,y4,score]
		bboxes.append(bbox)
	return bboxes

def apply_quad_nms(bboxes, overlap_threshold):
	dt_lines = sorted(bboxes, key=lambda x:-float(x[8]))
	nms_flag = nms(dt_lines, overlap_threshold)
	results=[]
	for k,dt in enumerate(dt_lines):
		if nms_flag[k]:
			if dt not in results:
				results.append(dt)
	return results

def save_and_visu(image, results, config,img_name):
	#img_name = img_path.split('/')[-1]
	det_save_path=os.path.join(config['det_save_dir'], img_name.split('.')[0]+'.txt')
	det_fid = open(det_save_path, 'wt')

	for result in results:
		score = result[-1]
		x1 = result[0]
		y1 = result[1]
		x2 = result[2]
		y2 = result[3]
		x3 = result[4]
		y3 = result[5]
		x4 = result[6]
		y4 = result[7]
		result_str=str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+str(x3)+','+str(y3)+','+str(x4)+','+str(y4)+','+str(score)+'\r\n'
		det_fid.write(result_str)
		if config['visu_detection']:
			quad = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
			quad = quad.reshape(-1,1,2)
			cv2.polylines(image, [quad], True, (0,0,255),3)

	det_fid.close()
	if config['visu_detection']:
                #print(img_name)
		cv2.imwrite(config['det_visu_dir']+img_name,np.uint8(image))

def detection(config,net,img_byte, width, height, img_num):
	#config = get_config();	

	#read model architecture and trained model's weights
	#net=prepare_network(config)

	#print('net preparation finished')

	#define image transformers
	transformer = caffe.io.Transformer({'data': (1,3,height, width)})
	transformer.set_transpose('data', (2, 0, 1))
	#transformer.set_mean('data', np.array([104,117,123])) # mean pixel
	#transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
	#transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
	net.blobs['data'].reshape(1,3,height, width)

	#Reading image paths
	#test_img_paths=[img_path for img_path in glob.glob(os.path.join(config['img_dir'],'*bmp'))]

	#if len(test_img_paths)== 0:
	#	print('Error: path error')

	#Making predictions
	img_name = str(img_num)+".bmp"
	print(img_name)		
		
	#img =np.float32(cv2.imread(img_path,cv2.IMREAD_COLOR))
	#img_path = test_img_paths[0]
	#img = skimage.img_as_float(skimage.io.imread(img_path, as_grey=False)).astype(np.float32) 
		
	img = np.frombuffer(img_byte,dtype='uint8')
	img = np.reshape(img,(height,width,3))

	img = np.float32(img)
	transformed_img = transformer.preprocess('data', img)
	net.blobs['data'].data[...] = transformed_img
	
	img_height, img_width, channels=img.shape
	  
	detections = net.forward()['detection_out']
	
	# Parse the outputs.
	bboxes = extract_detections(detections, config['det_score_threshold'], img_height, img_width)
 
	# apply non-maximum suppression
	results = apply_quad_nms(bboxes, config['overlap_threshold'])
	save_and_visu(img, results, config,img_name)
	return 1
