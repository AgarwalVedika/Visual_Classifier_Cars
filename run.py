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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
path3 = '/databaseForVedika/actual/'
listing3 = os.listdir(path3)
results_TP = open("new_TP.txt",'a') 
results_FP = open("new_FP.txt",'a')   
#### true boxes marked  
file1 = '/databaseForVedika/trackingData.txt'
data = np.genfromtxt(file1, comments = '#FRAME')[0:]
label = np.array([x[2:6] for x in data])

### FOR PLOTTING ###
y_axis = []
x_axis = []
the_ratio = []
def frange(theta):
	i = theta[0]  ##start
	while i < theta[1]:  ##stop
		yield i
		i += theta[2] ###step

#### adjust thresholds for theta1 and theta2: already used is the most optimum range observed
for theta1 in frange([0,4,2]):				
	for theta2 in frange([14,26,2]):		  ##### replace the equivalent number here							
		print theta1,theta2

		path4 = "/output/images_56/varying both_withNMS" + str(theta1) + "_" + str(theta2) + "/"  ### TP OUTPUT images 
		if not os.path.exists(path4):
				os.makedirs(path4)

		sum_detected_theta = 0;     ### if you want to calculate number of detected boxes for a particular theta for entire sequence of images
		TP_theta = 0
		FP_theta = 0
		sum_init = 0   ### for TP denominator
		i = 0
		for file2 in sorted(listing3):
			print file2
			image = cv2.imread(path3 + file2) 	## input image

			sum_overall = 0
			sum_overall_with_NMS = 0

			boxes = []  ## final NMS

			### Calculating gradients ####	
			img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			img_gray = cv2.medianBlur(img_gray, 5)
			sobelyvert = cv2.Sobel(img_gray, cv2.CV_32F, 1 , 0, ksize=5)/255.
			sobelyvert1 = sobelyvert*sobelyvert
			blank_image = np.zeros((img_gray.shape[0], img_gray.shape[1]), np.float32)
			blank_image_h = np.zeros((img_gray.shape[0], img_gray.shape[1]), np.float32)
			sobelyhor = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=5)/255.
			sobelyhor1 = sobelyhor*sobelyhor
			###### INTEGRAL IMAGES #####
			for x in xrange(0, img_gray.shape[1]):
				for y in xrange(0, img_gray.shape[0]):
					if y == 0:
						blank_image[y,x] =  sobelyvert1[y,x] 
						blank_image_h[y,x] =  sobelyhor1[y,x] 
					else:	
						blank_image[y,x] =  sobelyvert1[y,x] + blank_image[y-1,x]
						blank_image_h[y,x] =  sobelyhor1[y,x] + blank_image[y,x-1]	

			vector = np.zeros(int(image.shape[1]))
			vector_h = np.zeros(int(image.shape[1]))	


			for winW in xrange(30,100,10):									#### ADAPTIVE WINDOW SIZING ########
				boundingBoxes = []  		 #### for NMS
				stepSize= int(winW/3)
				winH = winW
				windowSize=(winW, winH)
				#print windowSize
		

				image = cv2.imread(path3 + file2) 	## input image		 ########### YOU HAVE TO CALL THE IMAGE OVER HERE ##########
				sum_detected = 0;   ### if you want to calculate number of detected boxes in image
		
				#### slide a window across the image ####
				for y1 in xrange(int(image.shape[0]*5/12), image.shape[0]- windowSize[0], stepSize):					
					y2 = y1 + winH 
					y_mid = int((y1+y2)/2)	
					vector = blank_image[y2,:] - blank_image[y_mid,:]				 ### tested before; use lower half
					vector_h = blank_image_h[y2,:] - blank_image_h[y_mid,:]           ######## HOR_GRAD summing over y; lower half #### CHANGE!!!		
					for x1 in xrange(0, image.shape[1]- windowSize[1], stepSize):
						# yield the current window
						x2 = x1 + winW	
						x_l = int (x1 + ((x2-x1+1)/3))
						x_r = int (x2 - ((x2-x1+1)/3))
						area = (x2 - x1 + 1) * (y2 - y1 + 1)
						mean_centre = (sum(vector[x_l:x_r]))/(x_r-x_l)		
						sum_hor_grad = sum(vector_h[x1:x2]) 							####### HOR_GRAD sum over x; entire window															
						max_left = np.amax(vector[x1:x_l])
						max_right = np.amax(vector[x_r:x2]) 
						if sum_hor_grad/area > theta1 :			### 3 gives 99% TP	; 2 gives 100%
							if max_left > mean_centre and max_right > mean_centre and np.absolute(max_left-max_right)<28*winH:												
								if ((max_left > theta2*winH and max_right > theta2*winH)or (max_left > theta2*winH and max_right > theta2*winH)): 						
									sum_detected = sum_detected + 1	
									#cv2.line(image, (x,y1), (x, int(y1-max_left/100)), (0,0,255),1)
									boundingBoxes.append((x1,y1,x2,y2,sum_hor_grad))          		#### for NMS
									#boxes.append((x1,y1,x2,y2,sum_hor_grad))
				#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!##########check from here ##########			
				
				boundingBoxes = np.array(boundingBoxes)   
		
				#### for NMS (for every window size)															
				# perform non-maximum suppression on the bounding boxes				
				pick = nms.non_max_suppression_fast(boundingBoxes, 0.3)				
							
				for (startX, startY, endX, endY, sum_hor_grad) in pick:	
					#cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 1)						
					boxes.append((startX, startY, endX, endY, sum_hor_grad))
				#print sum_detected
				#print "%d after applying NMS" % (len(pick))
				
	
			boxes1 = np.array(boxes)   							
			#print boxes

			# loop over the picked bounding boxes and draw them					
			for (startX, startY, endX, endY, sum_hor_grad) in boxes:							
				cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 1)				#### red
	
			'''
			###final NMS: first stage
			# perform non-maximum suppression on the  boxes				
			pick1 = nms_second.non_max_suppression_fast(boxes, 0.3)				
			#print pick1.shape

			## final NMS second stage							
			# perform non-maximum suppression on the  boxes				
			pick2 = nms_second.non_max_suppression_fast(pick1, 0.3)				
			#print pick2.shape

			# loop over the picked bounding boxes and draw them					
			for (startX, startY, endX, endY, sum_hor_grad) in pick2:							
				cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)				#### red
			''' 


			### if tp is stored for every frame in separate text file with an equivalent name
			'''
			###TP/FP COMPARISON ###
			ra = file2.rstrip('.png')
			rb = ra + '.txt'
		   	text = open(path5 + rb,'r') 				
			detetcions = text.readline()
			detections = eval(detetcions)
			#print detections		

			num_boxes_initial = np.shape(detections)[0]
			#print num_boxes_initial	
			sum_init = num_boxes_initial + sum_init
			#print sum_init
			'''
	
			### if labels are store for all the cars as shown below   #label = np.array([x[1:5] for x in data])   #### assumption: one car per frame
			'''>>> label
			array([[   2.        ,  237.42300415,  232.21200562,  261.96499634],
				   [   2.        ,  237.37399292,  230.53700256,  262.38400269],
				   [  11.        ,  237.27999878,  297.32299805,  268.60101318],
				   [  13.        ,  236.93200684,  234.80999756,  261.22698975]])'''
	
			sum_init = np.shape(label)[0]
			detections = label[i]
			i = i+1
	
			TP = 0
			FP = 0

			detections = detections.astype("int")
			#print detections
			#for a, b, c, d in detections: 
			score = 0
			x11 = detections[1] -5  
			y11 = detections[0] -5
			x21 = detections[3] +5
			y21 = detections[2] +5
			'''
			if len(file2) == 14:			
				x11 = a-5
				y11 = b
				x21 = c+5
				y21 = d+5
			elif len(file2) == 15: 
				x21 = (image.shape[1]-1) - (a-5)
				y11 = b
				x11 = (image.shape[1]-1) - (c+5)
				y21 = d+5
			'''
			cv2.rectangle(image, (x11, y11), (x21, y21), (255,255,255), 4)  #white box
	
			true_area = (x21 - x11 + 1) * (y21 - y11 + 1)
			for (x1, y1, x2, y2, s1) in boxes:
				area = (x2 - x1 + 1) * (y2 - y1 + 1)

				# find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
				xx1 = np.maximum(x11,x1)
				yy1 = np.maximum(y11,y1)
				xx2 = np.minimum(x21,x2)
				yy2 = np.minimum(y21,y2)

				# compute the width and height of the bounding box
				w = np.maximum(0, xx2 - xx1 + 1)			
				h = np.maximum(0, yy2 - yy1 + 1)	

				overlap = (w * h) / (area + true_area - (w*h))
				#print overlap1
		
				if overlap > 0.5:
					cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 4)  #blue box	
					cv2.rectangle(image, (x11, y11), (x21, y21), (0,255,0), 4) ## making the white: green				
					score = score +1

			if score >0:
				TP = TP+1
			'''
			if overlap < 0.5:
				#cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 4)  #blue box	
				#cv2.rectangle(image, (x11, y11), (x21, y21), (0,255,0), 4) ## making the white: green				
				#print ( (winH, winW), sum_hor_grad/area, max_left, max_right, mean_centre)				
				results_TP.write("{0},{1},{2},{3},{4} /n".format((winH, winW), sum_hor_grad/area, max_left, max_right, mean_centre))
			'''
			FP = boxes1.shape[0] - TP
			#print TP
			#print FP
			#print theta

			#results_TP1.write("{1}, ".format(theta, TP))
			#results_FP1.write("{1}, ".format(theta, FP))

			#cv2.imshow('image' ,image)
			#cv2.waitKey()
			cv2.imwrite(path4+ file2, image) 

			TP_theta = TP_theta + TP
			FP_theta = FP_theta + FP	
		
		reqd_ratio = TP_theta/FP_theta
		print (TP_theta/sum_init)	
		print (FP_theta/60)
		print (reqd_ratio)
		results_TP.write("({0},{1})-{2},  ".format(theta1, theta2, TP_theta/sum_init))
		results_FP.write("({0},{1})-{2}, ".format(theta1, theta2, FP_theta/130))	
		y_axis.append(theta2) 
		x_axis.append(theta1) 
		the_ratio.append(reqd_ratio) 

print ("theta1")
print (x_axis)
print("theta2")
print(y_axis)
print("the_ratio")
print(the_ratio)

#### FOR PLOTTING PURPOSE: vary theta1 and theta2; the TP/FP will be annotated
the_list = []
for i in the_ratio:
	b = str(i)
	print b[0:8]
	the_list.append(eval(b[0:8]))
#print the_list

y_axis = theta2
x_axis = theta1

#print len(y_axis)
#print len(x_axis)
#print len(the_ratio11)

fig, ax = plt.subplots()
ax.scatter(x_axis, y_axis)
for i, txt in enumerate(the_list):
    ax.annotate(txt, (x_axis[i],y_axis[i]))
plt.axis([0, 4, 12, 26])   ### ADJUST THE SCALE ACCODIGNLY; THIS IS FOR THE OPTIMUM RANGE
plt.xlabel('theta1')
plt.ylabel('theta2')
plt.title('both thresholds varied; TP/FP indicated on the graph')
plt.grid()
plt.show()

results_TP.close
results_FP.close

cv2.destroyAllWindows()
