#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 22:05:28 2017

@author: dinesh
"""

import cv2
import glob
import numpy as np
import os
if __name__ == "__main__":
    
    #### read image file names
    base_dir = '/home/dinesh/CarCrash/data/CarCrash/Test'
    base_dir = '/home/dinesh/CarCrash/data/CarCrash/Cleaned2/'
    inputs = range(26)
    images = []
    masks = []
    for i,num in enumerate(inputs):
        image_list = sorted(glob.glob(base_dir + '/'+ str(i) + '/images/*.png'))
        mask_list = sorted(glob.glob(base_dir + '/'+ str(i) + '/mask/*'))
        #print(mask_list)
        if len(mask_list) != len(image_list):
            print('mask Not available')
        images.append(image_list)
        masks.append(mask_list)
        if not os.path.exists(base_dir + '/'+ str(i) + '/mask_cleaned/'):
                os.makedirs(base_dir + '/'+ str(i) + '/mask_cleaned/')
        if not os.path.exists(base_dir + '/' + '/car1/'):
                os.makedirs(base_dir + '/' + '/car1/')    
        if not os.path.exists(base_dir + '/' + '/car2/'):
                os.makedirs(base_dir + '/' + '/car2/')       
    #### read image file names
    num_of_cars = 2
    for camera,files in enumerate(masks):
       for num,filenames in enumerate(files):
            img_path = images[camera][num]
            print(filenames)
            img = cv2.imread(filenames,0)
            img_real = cv2.imread(img_path)
            assert(filenames.split('/')[-1]==img_path.split('/')[-1])
            ret, markers = cv2.connectedComponents(img)
            final_mask = markers
            car_count = 1
            final_img = img_real*0
            final_mask_img = img*0
            if not os.path.exists(base_dir + '/'+ str(camera) + '/car1/'):
                os.makedirs(base_dir + '/'+ str(camera) + '/car1/')
            cv2.imwrite(base_dir +  str(camera) + '/car1/' + filenames.split('/')[-1] , final_img)
            for labels,num in enumerate(np.unique(final_mask)):
                if np.count_nonzero(final_mask==num) > 30000 and num != 0:
                    seg_img = img_real[np.where(final_mask==num)]
                    hist = cv2.calcHist([seg_img],[0],None,[3],[0,256])
                    
                    ## labelling blue car
                    print(hist)
                    if np.argmax(hist) == 0 and hist[0]/np.sum(hist) >0.90:
                        final_mask_img[np.where(final_mask==num)] = 100
                        final_img = final_img*0
                        final_img[np.where(final_mask==num)] = img_real[np.where(final_mask==num)]
                        cv2.imwrite(base_dir + '/car1/' + filenames.split('/')[-1].split('.')[0] + '_'  + str(camera) + '.png'  , final_img)
                        cv2.imwrite(base_dir +  str(camera) + '/car1/' + filenames.split('/')[-1] , final_img)
                    elif hist[0]/np.sum(hist) < 0.50 :
                        final_mask_img[np.where(final_mask==num)] = 50
                        final_img = final_img*0
                        final_img[np.where(final_mask==num)] = img_real[np.where(final_mask==num)]
                        cv2.imwrite(base_dir + '/car2/' + filenames.split('/')[-1].split('.')[0] + '_'  + str(camera) + '.png'  , final_img)
                    else:
                        final_mask_img[np.where(final_mask==num)] = 255

            cv2.imwrite(filenames.replace("mask","mask_cleaned"), final_mask_img)
            print(filenames)
