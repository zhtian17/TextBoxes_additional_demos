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

import subprocess

import glob
import cv2
import skimage.io

def get_config():
	config = {
		'model_def' : '../../models/deploy.prototxt',
		'model_weights' : '../../models/model_icdar15.caffemodel',
		#'img_dir' : '../../demo_images/test/',
		'det_visu_dir' : '../../demo_images/test_result_img/',
		'det_save_dir' : '../../demo_images/test_result_txt/',
		'crop_dir' : '../../demo_images/crops/',
		'lexicon_path' : '../../crnn/data/icdar_generic_lexicon.txt',
		'use_lexcion' : True,
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


def extract_detections(detections, idx, det_score_threshold, image_height, image_width):
	#for idx in range(0,img_num)
	detections_ = detections[0, 0]
	img_idx = [i for i,det in enumerate(detections_) if det[0] == idx]
	detection_single = detections_[img_idx]
	#detection_single = detections
	det_conf = detection_single[:,2]
	det_x1 = detection_single[:,7]
	det_y1 = detection_single[:,8]
	det_x2 = detection_single[:,9]
	det_y2 = detection_single[:,10]
	det_x3 = detection_single[:,11]
	det_y3 = detection_single[:,12]
	det_x4 = detection_single[:,13]
	det_y4 = detection_single[:,14]
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
	print(det_save_path)
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

def detection(config,net,byte_imgs, width, height, img_names, img_num):
	#reshape the data layer
	net.blobs['data'].reshape(img_num, 3, height, width)
	#Process images
	imgs = np.array([np.frombuffer(byte_imgs[i],dtype='uint8') for i in range(img_num)])
	imgs = np.reshape(imgs, (img_num, height, width, 3))
	imgs_processed = np.transpose(imgs, (0, 3, 1, 2))
	imgs_processed = np.float32(imgs_processed)

	net.blobs['data'].data[...] = imgs_processed
	detections = net.forward()['detection_out']

	# Parse the outputs
	for idx in range(img_num):
		bboxes = extract_detections(detections, idx, config['det_score_threshold'], height, width)
		# apply non-maximum suppression
		results = apply_quad_nms(bboxes, config['overlap_threshold'])
		save_and_visu(imgs[idx], results, config, img_names[idx])
	return 1
