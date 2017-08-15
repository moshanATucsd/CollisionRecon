#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 19:13:13 2017

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
#Folder = '/home/dinesh/CarCrash/data/test/'


    # front head lights
def label_bb(img,kp,bb,bb_loop,track):
    if class_name[bb_loop] == 'car':
        cv2.rectangle(img,(int(bb[bb_loop][0]),int(bb[bb_loop][1])),(int(bb[bb_loop][0] + bb[bb_loop][2]),int(bb[bb_loop][1]+bb[bb_loop][3])),(0,255,0), thickness=2)
        cv2.putText(img,'CAR '+ str(track[bb_loop]), (bb[bb_loop][0],bb[bb_loop][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        drawCar(img,kp,bb)
    if class_name[bb_loop] == 'person':
        cv2.putText(img,'PERSON '+ str(track[bb_loop]), (bb[bb_loop][0],bb[bb_loop][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
        cv2.rectangle(img,(int(bb[bb_loop][0]),int(bb[bb_loop][1])),(int(bb[bb_loop][0] + bb[bb_loop][2]),int(bb[bb_loop][1]+bb[bb_loop][3])),(255,0,0), thickness=2)
        drawPerson(img,kp,bb)
    if class_name[bb_loop] == 'truck':
        cv2.rectangle(img,(int(bb[bb_loop][0]),int(bb[bb_loop][1])),(int(bb[bb_loop][0] + bb[bb_loop][2]),int(bb[bb_loop][1]+bb[bb_loop][3])),(0,0,255), thickness=2)
        drawCar(img,kp,bb)
        cv2.putText(img,'TRUCK '+ str(track[bb_loop]), (bb[bb_loop][0],bb[bb_loop][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        #cv2.putText(img,'TRUCK '+ str(track[bb_loop]), (bb[bb_loop][0]+3,bb[bb_loop][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
    if class_name[bb_loop] == 'bus':
        cv2.rectangle(img,(int(bb[bb_loop][0]),int(bb[bb_loop][1])),(int(bb[bb_loop][0] + bb[bb_loop][2]),int(bb[bb_loop][1]+bb[bb_loop][3])),(255,0,255), thickness=2)
        cv2.putText(img,'BUS '+ str(track[bb_loop]), (bb[bb_loop][0],bb[bb_loop][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255),2)
        drawCar(img,kp,bb)

def getRGBfromI(RGBint):
    blue =  RGBint & 255
    green = (RGBint >> 8) & 255
    red =   (RGBint >> 16) & 255
    return [red, green, blue]

for main_loop in range(21,21):
    filenames = sorted(glob.glob(Folder + str(main_loop-1) + '/keypoints_tracked/*.txt'))
    colormap = np.random.randint(100,2147483646,size = 100000)
    for index,name in enumerate(filenames):
        bb = []
        points = []
        class_name = []
        track = []
        img_name = name.split('keypoints_tracked')[0] + name.split('keypoints_tracked')[1].split('.txt')[0]
        img_original = cv2.imread(img_name)
        img_instance_segment = cv2.imread(img_name.replace('//','/labelled/'))
        with open(name) as f:
            lines = f.readlines()
        for line in lines:
            bb.append(np.array(line.split(',')[1:5]).astype(np.int))
            points.append(np.array(line.split(',')[5:-1]).astype(np.float))
            class_name.append(line.split(',')[-1].split('\n')[0])   
            track.append(np.array(line.split(',')[0]).astype(np.int))

        for inde,bounding in enumerate(bb):
            img_crop = img_instance_segment[bounding[1]:bounding[1]+bounding[3],bounding[0]:bounding[0]+bounding[2]]
            gray_image = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            elements,count = np.unique(gray_image, return_counts=True)
            count_max = 0
            ele = elements[-1]
            for i,a in enumerate(elements):
                if i == 0:
                    continue
                if count[i] > count_max:
                    ele = a
                    count_max = count[i]
            #print(ele)
            color = img_crop[np.where(gray_image==ele)]
            #print(color[0])
            #print(getRGBfromI(colormap[track[inde]]))
            if ele == 0:
                continue
            img_instance_segment[np.where((img_instance_segment==color[0]).all(axis=2))] = getRGBfromI(colormap[track[inde]])
        
        #cv2.imwrite('a.png',img_instance_segment)
        #sada
        img = img_instance_segment*0#img_original*0
        img_final = img_original
        print(img_name)
        
        for bb_loop,bb_num in enumerate(bb):
            points_array = np.array(points[bb_loop])#.splitlines()[0].split(','))
            points_arranged = points_array.reshape(int(len(points_array)/3),3)
            kp = points_arranged[:,0:3]
            kp = np.round(kp.astype(np.float)).astype(np.int)
            kp[:,0] = bb[bb_loop][0] + kp[:,0]*(bb[bb_loop][2]/64)
            kp[:,1] = bb[bb_loop][1] + kp[:,1]*(bb[bb_loop][3]/64)
            kp[:,2] = np.round(points_arranged[:,2].astype(np.float)*100).astype(np.int)
            label_bb(img,kp,bb,bb_loop,track)
            
            label_bb(img_original,kp,bb,bb_loop,track)
        #img_original
        cv2.addWeighted(img_original, 1, img_instance_segment, 0.5, 0, img_final)
        cv2.addWeighted(img, 1, img_instance_segment, 0.5, 0, img)
        #cv2.addWeighted(img_original, 1, img, 10, 0, img_final)
        
        #cv2.addWeighted(img_final, 1, img_final, 1, 0, img_final)
        cv2.imwrite(img_name.replace('//','/labelled_image/'), img_final)
        cv2.imwrite(img_name.replace('//','/labelled_image/b_'), img)
        #asas
