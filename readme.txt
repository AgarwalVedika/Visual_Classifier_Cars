Author: Vedika Agarwal
Supervisor: Prof. Alexander Gepperth

Project: Visual Classifier for Cars

This folder contains the codes and the results for the prelimniary stage of the built classifier. The algorithm uses a sliding window based approach over a range of sizes. The preliminary detection utlizies a linear combination of low-level features (edges/gradients); exploits the information of the distance of the car (from an estimated road plane) and then uses a non maximal suprresion technique in order to reduce the number of repeated detections: thus significantly reducing the number of false positives.

run.py
You can vary the thresholds: theta1 and theta2 in order to see its effect on TP/FP
Observations: (2,18) serves as the ideal combination of (theta1,theta2) with  a TP/FP = 0.012574 without NMS and 0.018563 with NMS
			  (0,20) doesn't look bad either with a TP/FP = 0.012493 without NMS and 0.018798 with NMS

You can comment out the NMS portion, it's included by default. Moreover the furher stages could be included too. The detector folder is meant for the imports of nms codes. Nms_second also incorportaes the gradient version alone with the overalpped area while doing the suppression.

Supplementary_material
Cobtains codes and some results which were used to visualze the effect of gradients on the dataset and accordingly decide the criteria.
