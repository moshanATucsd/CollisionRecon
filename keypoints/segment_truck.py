#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:52:59 2017

@author: dinesh
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
from image_plot import *



Folder = '/home/dinesh/CarCrash/data/Fifth/'
#Folder = '/home/dinesh/CarCrash/data/CarCrash/Cleaned/'
#Folder = '/home/dinesh/CarCrash/data/syn/'
#Folder = '/home/dinesh/CarCrash/data/Kitti_1/'


    # front head lights
def label_bb(img,kp,bb,bb_loop):
    if class_name[bb_loop] == 'car':
        cv2.rectangle(img,(w_x,w_x_end),(w_y,w_y_end),(0,255,0))
        drawCar(img,kp,bb)
    if class_name[bb_loop] == 'person':
        cv2.rectangle(img,(w_x,w_x_end),(w_y,w_y_end),(255,0,0))
        drawPerson(img,kp,bb)
    if class_name[bb_loop] == 'truck':
        cv2.rectangle(img,(w_x,w_x_end),(w_y,w_y_end),(0,0,255))
        drawCar(img,kp,bb)
        imgcrop = img[w_x:w_x_end,w_y:w_y_end]
        cv2.imwrite()
    if class_name[bb_loop] == 'bus':
        cv2.rectangle(img,(w_x,w_x_end),(w_y,w_y_end),(255,0,255))
        drawCar(img,kp,bb)

count_bus = 0
count_truck = 0

for main_loop in range(1,21):
    filenames = sorted(glob.glob(Folder + str(main_loop-1) + '/keypoints_txt/*.txt'))
    for index,name in enumerate(filenames):
        if index%50 != 0:
            continue
        bb = []
        points = []
        class_name = []
        img_name = name.split('keypoints_txt')[0] + name.split('keypoints_txt')[1].split('.txt')[0]
        img_original = cv2.imread(img_name)
        img_instance_segment = cv2.imread(img_name.replace('//','/labelled/'))
        img = img_instance_segment*0#img_original*0
        img_final = img_original
        print(img_name)
        with open(name) as f:
            lines = f.readlines()
        for line in lines:
            bb.append(np.array(line.split(',')[1:5]).astype(np.float))
            points.append(np.array(line.split(',')[5:-1]).astype(np.float))
            class_name.append(line.split(',')[-1].split('\n')[0]) 
        for bb_loop,bb_num in enumerate(bb):
            points_array = np.array(points[bb_loop])#.splitlines()[0].split(','))
            points_arranged = points_array.reshape(int(len(points_array)/3),3)
            kp = points_arranged[:,0:3]
            kp = np.round(kp.astype(np.float)).astype(np.int)
            kp[:,0] = bb[bb_loop][0] + kp[:,0]*(bb[bb_loop][2]/64)
            kp[:,1] = bb[bb_loop][1] + kp[:,1]*(bb[bb_loop][3]/64)
            kp[:,2] = np.round(points_arranged[:,2].astype(np.float)*100).astype(np.int)
            w_x = int(bb[bb_loop][0])
            w_x_end =int(bb[bb_loop][0] + bb[bb_loop][2])
            w_y = int(bb[bb_loop][1])
            w_y_end =int(bb[bb_loop][1] + bb[bb_loop][3])
            if class_name[bb_loop] == 'truck':
                print(class_name[bb_loop])
                imgcrop = img_original[w_y:w_y_end,w_x:w_x_end]
                cv2.imwrite(str(count_truck)+'.png',imgcrop)
                count_truck +=1
            if class_name[bb_loop] == 'bus':
                print(class_name[bb_loop])
                imgcrop = img_original[w_y:w_y_end,w_x:w_x_end]
                cv2.imwrite(str(count_truck)+'.png',imgcrop)
                count_truck +=1