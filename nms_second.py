#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import the necessary packages
import numpy as np
''' 
>>> boxes = np.array([
	(12, 84, 140, 212),
	(24, 84, 152, 212),
	(36, 84, 164, 212),
	(12, 96, 140, 224),
	(24, 96, 152, 224),
	(24, 108, 152, 236)])
s1 = np.array([534, 123, 787, 765, 234, 345])
s1 = s1.astype("float")
'''
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]				#array([  12.,   24.,   36.,   12.,   24.,   24.])
	y1 = boxes[:,1]				#array([  84.,   84.,   84.,   96.,   96.,  108.])
	x2 = boxes[:,2]				#array([ 140.,  152.,  164.,  140.,  152.,  152.])
	y2 = boxes[:,3]				#array([ 212.,  212.,  212.,  224.,  224.,  236.])
 	s1 = boxes[:,4]				#array([ 534.,  123.,  787.,  765.,  234.,  345.])

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)		#array([ 16641.,  16641.,  16641.,  16641.,  16641.,  16641.])
	idxs = np.argsort(y2)						#array([0, 1, 2, 3, 4, 5])
	
	#criterion1: compute the ratio
	cr1= s1/area								#array([ 0.03208942,  0.00739138,  0.04729283,  0.0459708 ,  0.01406165,  0.02073193])
	#print cr1
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1    #5
		i = idxs[last]			#5
		pick.append(i) 		
		
		#cc1 = np.maximum(cr1[i], cr1[idxs[:last]])  #array([ 0.03208942,  0.02073193,  0.04729283,  0.0459708 ,  0.02073193])

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])    #array([  24.,   24.,   36. ,  24.,   24.])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])	#array([ 108.,  108.,  108.,  108.,  108.])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])	#array([ 152.,  152.,  164.,  152.,  152.])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])	#array([ 236.,  236.,  236.,  236.,  236.])
 		
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)			#array([ 129.,  129.,  129.,  129.,  129.])
		h = np.maximum(0, yy2 - yy1 + 1)			#array([ 129.,  129.,  129.,  129.,  129.])
		'''
		## overlap check: if the boxes intersect or not: xx1<vx2i and yy1<vy2i 
		vx2i = np.repeat(x2[i],last)				#array([ 152.,  152.,  152.,  152.,  152.])
		vy2i = np.repeat(y2[i],last)				#array([ 236.,  236.,  236.,  236.,  236.])
		a = xx1<vx2i
		b = yy1<vy2i
		intersect = a&b.astype(int)
 		print intersect
		'''
		## cr1 check: boxes compared for higher sum_hor_grad/area
		vcr1 = 	np.repeat(cr1[i],last)	
		c = vcr1>cr1[:last]                          #array([False,  True, False, False,  True], dtype=bool)
		c1 = c.astype(int)
		# compute the ratio of overlap
		overlap1 = (w * h) / area[idxs[:last]]      #array([ 1.,  1.,  1.,  1.,  1.])   ## area[idxs[:last]] = array([ 16641.,  16641.,  16641.,  16641.,  16641.])
		#overlap = overlap1*c1*intersect			 #array([ 1.,  1.,  1.,  1.,  1.])
		overlap = overlap1*c1
		#print overlap1
		#print overlap
		#print overlap 

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))		#array([], dtype=int64)   ## for last = 5; all the indices are deleted: empty array: overlap>1
																									#np.where(overlap > 0.3)  (array([0, 1, 2, 3, 4]),)
						 																			#np.concatenate(([last],np.where(overlap > 0.3)[0])) = array([5, 0, 1, 2, 3, 4])

	## pick = 5; now idxs is empty; so no more pick appends; hence answer is box 5: 24.,108.,152.,236.
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")
