#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:18:16 2017

@author: dinesh
"""
import numpy as np
from read_write_data import *
from check_cameras import *
from python_ceres.ba import *
from Car_fit import *


Folder = '/home/dinesh/CarCrash/data/Fifth/'
#Folder = '/home/dinesh/CarCrash/data/syn/'


def tracklet_data(tracklet_num,camera,time,kp_track,all_time,all_cam,camera_index):
    
    new_time = time
    flag = 0
    while(1):
        new_time = new_time - 1
        synched_images = - diff - diff_ext + new_time  
        data = Read_keypoints(synched_images,Folder,leading_zeros)
        for ind,kp in enumerate(data[camera]):
            if len(kp) > 0:#print(kp)
                if len(kp[0]) > 0:
                    if tracklet_num == kp[0]:
                        flag = 0
                        break
                    else:
                        flag = 1
        if flag == 1:
            break
    
    
    
    while(1):
        new_time = new_time + 1
        synched_images = - diff - diff_ext + new_time  
        data = Read_keypoints(synched_images,Folder,leading_zeros)
        for ind,kp in enumerate(data[camera]):
            if len(kp[0]) > 0:
                if tracklet_num == kp[0]:
                    flag = 0
                    keypoints,bb = data_to_kp(kp)
                    kp_track.append(keypoints)
                    camera_index.append(synched_images[camera])
                    all_time.append(new_time)
                    all_cam.append(camera)
                    break
                else:
                    flag = 1
            
        if flag == 1:
            break
    return kp_track

def get_tracklets(tracklet,cam_index,corr,K,time):
    kp_track = []
    all_cam = []
    all_time = []
    camera_index = []
    for a,val in enumerate(corr):
        tracklet_data(tracklet[val],cam_index[val],time,kp_track,all_time,all_cam,camera_index)
    return kp_track,all_time,all_cam,camera_index

def video_data_ceres(kp_track,all_time,all_cam,camera_index,RT,K):
    point_2d = []
    K_ceres = []
    RT_ceres = []
    correspondence = []
    car_time = []
    cam_ind = []
    for loop,loop_num in enumerate(camera_index):
        cam = all_cam[loop]
        Index_num = np.where(np.array(RT_index[cam]).astype(np.int) == loop_num)
        if len(Index_num[0]) == 1:
            RT_cam = RT[cam][int(Index_num[0])]
            K_cam = K[cam][int(Index_num[0])]
            for index,point in enumerate(kp_track[loop]):
                if point[2]<50 or index == 8 or index == 9:# or kk > 3:
                    continue
                K_ceres.append(K_cam)
                RT_ceres.append(RT_cam)
                point_2d = np.concatenate((point_2d,point[0:2]))
                correspondence.append(index)
                car_time.append(all_time[loop] - min(all_time))
                cam_ind.append(cam)
    return K_ceres,RT_ceres,point_2d,correspondence,car_time,cam_ind



def bundleAdjust_video(K_ceres,RT_ceres,point_2d,correspondence,car_time,cam_ind,scale_all,ind,car_rt_all):
    RT_transform = []
    keypoints = np.loadtxt('/home/dinesh/CarCrash/codes/CollisionRecon/keypoints/car_cad/car6_kp.txt', dtype='f', delimiter=',')
    scale = scale_all[ind]
    for a in range(max(car_time)):
            #RT_transform.append(np.identity(4))
            RT_transform.append(car_rt_all[ind])
                
    np.savez('car_video_test', K_ceres, RT_ceres, keypoints.reshape(len(keypoints)*3), point_2d,correspondence,scale,RT_transform,car_time,cam_ind)
    scale,RT_transform = car_fit_video_nview_with_ceres(K_ceres, RT_ceres,keypoints.reshape(len(keypoints)*3),point_2d,correspondence,scale,RT_transform,car_time)
    return scale, RT_transform



def video_carfit(time,RT,RS, RT_index,K, K_index,diff,diff_ext,video_window):
            
            
            synched_images = - diff + time - diff_ext
            synched_images_ext = - diff + time #+ diff_ext
        
            ## read keypoints
            data = Read_keypoints(synched_images,Folder,leading_zeros)
            all_bb,kp_all,cam_index,tracklet,RT_all,K_all = time_instance_data(data,synched_images_ext,RT,RT_index,K,K_index)
            Car_3d,correspondence_all,scale_all,car_rt_all = reconstruct_keypoints(all_bb,kp_all,cam_index,RT_all,K_all)       
            
            for ind,corr in enumerate(correspondence_all):
                kp_track,all_time,all_cam,camera_index = get_tracklets(tracklet,cam_index,corr,K,time)
                K_ceres,RT_ceres,point_2d,correspondence,car_time,cam_ind = video_data_ceres(kp_track,all_time,all_cam,camera_index,RT,K)

                scale, RT_transform_all = bundleAdjust_video(K_ceres,RT_ceres,point_2d,correspondence,car_time,cam_ind,scale_all,ind,car_rt_all)
                
                
                for inde,RT_transform in enumerate(RT_transform_all):
                    time_save = min(car_time) + inde
                    synched_images = - diff + time_save - diff_ext
                    data = Read_keypoints(synched_images,Folder,leading_zeros)
                    all_bb,kp_all,cam_index,tracklet,RT_all,K_all = time_instance_data(data,synched_images_ext,RT,RT_index,K,K_index)
                    
                    
                    keypoints = np.loadtxt('/home/dinesh/CarCrash/codes/CollisionRecon/keypoints/car_cad/car6_kp.txt', dtype='f', delimiter=',')
                    point_3d_car_kp_all_best =[]
                    for kk,loop in enumerate(keypoints):
                        keypnt = np.transpose(np.dot(RT_transform,np.transpose(np.append(keypoints[kk]*scale,1))))
                        #print(np.append(keypoints[kk]*scale,1))
                        #print(keypnt)
                        if kk == 8 or kk == 9:
                            point_3d_car_kp_all_best.append([])
                        else:
                            point_3d_car_kp_all_best.append(np.reshape(keypnt,[4,1]))
                    camera,correspondence_all = count_inliers_kp(all_bb,point_3d_car_kp_all_best,K_all,RT_all,kp_all,cam_index)
                    save_images(point_3d_car_kp_all_best,correspondence_all,cam_index,all_bb,K_all,RT_all,time,Folder,synched_images)

                    #print('here')
                    #print(scale,RT_transform)

                
                
            
            

num_cams = 21
leading_zeros = 5
video_window = 11
Rollingshutter = False 
RT,RS, RT_index,K, K_index,diff,diff_ext = Read_data(Folder,num_cams,Rollingshutter)
time = 630
while time < 10000:
    print('Checking for objects in time instance ',time)
    time = time + 1
    #image_carfit(time,RT,RS, RT_index,K, K_index,diff,diff_ext)
    video_carfit(time, RT, RS, RT_index, K, K_index, diff, diff_ext, video_window)
    #print(len(kp_track))


            
