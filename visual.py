#!/usr/bin/env python
# -*- coding: utf-8 -*-
##############~Imports~##############
from __future__ import division
from detector import nms
from detector import nms_second
import cv2
import glob
import os,sys
import re
import math
import numpy as np

#path3 = "/home/vedika/Downloads/detector/main/images/"            ### INPUT images  
path3 = "/home/vedika/Downloads/databaseForVedika/actual/"
listing3 = os.listdir(path3)
path4 = "/home/vedika/Downloads/detector/main/output/images_56/visualize_the_behaviour/" ### TP OUTPUT images 
if not os.path.exists(path4):
		os.makedirs(path4)
file1 = '/home/vedika/Downloads/databaseForVedika/trackingData.txt'
data = np.genfromtxt(file1, comments = '#FRAME')[0:]
label = np.array([x[2:6] for x in data])
results = open("max_l-max_r diff.txt",'a')
i = 0							
for file2 in sorted(listing3):
	#print file2
	image = cv2.imread(path3 + file2) 	## input image
	detections = label[i]
	i = i+1
	detections = detections.astype("int")
	x1 = detections[1] -5  
	y1 = detections[0] -5
	x2 = detections[3] +5
	y2 = detections[2] +5
	winH = y2-y1
	y_mid = int((y1+y2)/2)	
	x_l = int (x1 + ((x2-x1+1)/3))
	x_r = int (x2 - ((x2-x1+1)/3))
	array = np.zeros(int(x2-x1+1))
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	max_centre = 0
	max_left = 0
	max_right = 0
	sum_mean = 0
	### Calculating gradients ####	
	img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img_gray = cv2.medianBlur(img_gray, 5)
	sobelyvert = cv2.Sobel(img_gray, cv2.CV_32F, 1 , 0, ksize=5)/255.
	sobelyvert1 = sobelyvert*sobelyvert
	blank_image = np.zeros((img_gray.shape[0], img_gray.shape[1]), np.float32)
	blank_image_h = np.zeros((img_gray.shape[0], img_gray.shape[1]), np.float32)
	sobelyhor = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=5)/255.
	sobelyhor1 = sobelyhor*sobelyhor

	cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 1)   #red box
	for x in xrange(x1,x2):
		sum_vert_grad = 0
		for y in xrange(y_mid,y2):																		##### CHANGE 1 #########
			sum_vert_grad = sum_vert_grad + sobelyvert1[y,x]	
		array[x-x1]= sum_vert_grad
		cv2.line(image, (x,y1), (x, int(y1-array[x-x1]/100)), (0,255,0),1)	#green lines for all 
	#print array
	for x in xrange(x_l, x_r):
		#if array[x-x1] > max_centre:
		#	max_centre = array[x-x1]
		sum_mean = array[x-x1] + sum_mean
		mean_centre = sum_mean/(x_r-x_l)
	for x in xrange(x1,x_l):
		if array[x-x1] > max_left:
			max_left = array[x-x1]
		#if array[x-x1] < min_left:
		#	min_left = array[x-x1]
	for x in xrange(x_r,x2):
		if array[x-x1] > max_right:
			max_right = array[x-x1]
		#if array[x-x1] < min_right:
		#	min_right = array[x-x1]
	cv2.line(image, (x,y1), (x, int(y1-array[x-x1]/100)), (0,0,255),1)
	cv2.imwrite(path4+ file2, image) 
	results.write("{0},{1},{2}\n".format(((max_left-max_right)/winH),max_left/winH,max_right/winH))
results.close
cv2.destroyAllWindows()
