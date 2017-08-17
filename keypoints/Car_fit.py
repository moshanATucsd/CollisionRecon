#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:16:35 2017

@author: dinesh
"""

import numpy as np
from read_write_data import *
from check_cameras import *
from python_ceres.ba import *


num_cams = 21
leading_zeros = 5
video_window = 2
Rollingshutter = False 
Folder = '/home/dinesh/CarCrash/data/Fifth/'

def check_nviews_ransac(RT_all,K_all,all_bb,cam_index,kp_all,point_3d_kp_all):
    camera,cameras = count_inliers(all_bb,point_3d_kp_all,K_all,RT_all,kp_all,cam_index)
    num_added =  np.zeros(len(point_3d_kp_all))
    reproject_error_kp = 0
    point_3d_kp_eachview = []
    for l,cams_l in enumerate(cameras):
        for h,cams_h in enumerate(cameras):
            RT_1 = np.append(RT_all[cams_l],[0,0,0,1]).reshape(4,4)
            RT_2 = np.append(RT_all[cams_h],[0,0,0,1]).reshape(4,4)
            RT_12 = np.dot(np.linalg.inv(RT_1),RT_2)
            P_1 = np.dot(K_all[cams_l],  RT_all[cams_l]) 
            P_2 = np.dot(K_all[cams_h],  RT_all[cams_h])
            if np.mean(RT_12[0:3,3]) < 0.1:
                continue
            kp_1 =  kp_all[cams_l]
            kp_2 = kp_all[cams_h]
            point_3d_kp_eachview.append(triangulate_visible_keypoints(P_1,P_2,kp_1,kp_2))
    
    point_3d_kp_all_best = point_3d_kp_all
    for kk,loop in enumerate(point_3d_kp_all_best):
        count_best = 10000
        for k,p_3d_kp in enumerate(point_3d_kp_eachview):
            #count = count_inliers_point(all_bb,point_3d,K_all,RT_all)
            point_3d = p_3d_kp[kk]
            count = reproj_inliers_point(all_bb, point_3d, K_all, RT_all, kp_all, kk, cameras)
            if count < count_best:
                point_3d_kp_all_best[kk] = point_3d
                count_best = count


#    keypoints = np.loadtxt('/home/dinesh/CarCrash/codes/CollisionRecon/keypoints/car_cad/car6_kp.txt', dtype='f', delimiter=',')
#    scale,RT_transform = scale_transformation(point_3d_kp_all,keypoints)
#    point_3d_car_kp_all_best =[]
#    for kk,loop in enumerate(keypoints):
#        keypnt = np.transpose(np.dot(RT_transform,np.transpose(np.append(keypoints[kk],1))))
#        if kk == 8 or kk == 9:
#            point_3d_car_kp_all_best.append([])It's showing "II Sem Fees*", is that an issue?
#        else:
#            point_3d_car_kp_all_best.append(np.reshape(keypnt,[4,1]))
        
    #print(keypoints)
    #print(point_3d_car_kp_all_best)
    camera,cameras = count_inliers(all_bb,point_3d_kp_all_best,K_all,RT_all,kp_all,cam_index)
    return len(np.unique(camera)),cameras,point_3d_kp_all_best,reproject_error_kp


def check_nviews_nonlinear(RT_all,K_all,all_bb,cam_index,kp_all,point_3d_kp_all):
    camera,cameras = count_inliers(all_bb,point_3d_kp_all,K_all,RT_all,kp_all,cam_index)
    num_added =  np.zeros(len(point_3d_kp_all))
    reproject_error_kp = 0
    point_3d_kp_eachview = []
    
    point_3d_kp_all_best = point_3d_kp_all
    for l,cams_l in enumerate(cameras):
        for h,cams_h in enumerate(cameras):
            RT_1 = np.append(RT_all[cams_l],[0,0,0,1]).reshape(4,4)
            RT_2 = np.append(RT_all[cams_h],[0,0,0,1]).reshape(4,4)
            RT_12 = np.dot(np.linalg.inv(RT_1),RT_2)
            P_1 = np.dot(K_all[cams_l],  RT_all[cams_l]) 
            P_2 = np.dot(K_all[cams_h],  RT_all[cams_h])
            if np.mean(RT_12[0:3,3]) < 0.1:
                continue
            kp_1 =  kp_all[cams_l]
            kp_2 = kp_all[cams_h]
            point_3d_kp_eachview.append(triangulate_visible_keypoints(P_1,P_2,kp_1,kp_2))
    
    point_3d_kp_all_best = point_3d_kp_all
    for kk,loop in enumerate(point_3d_kp_all_best):
        count_best = 10000
        for k,p_3d_kp in enumerate(point_3d_kp_eachview):
            #count = count_inliers_point(all_bb,point_3d,K_all,RT_all)
            point_3d = p_3d_kp[kk]
            count = reproj_inliers_point(all_bb,point_3d,K_all,RT_all,kp_all,kk,cameras)
            if count < count_best:
                point_3d_kp_all_best[kk] = point_3d
                count_best = count
    
    for kk,loop in enumerate(point_3d_kp_all_best):
        RT_ceres = []
        K_ceres = []
        point_2d = []
        for l,cams_l in enumerate(cameras):
            if kp_all[cams_l][kk,2]>80:# or kk > 3:
                continue
            RT_ceres.append(RT_all[cams_l])
            K_ceres.append(K_all[cams_l])

            #print(kp_all[cams_l][kk,0:2])
            point_2d = np.concatenate((point_2d,kp_all[cams_l][kk,0:2]))
        point_3d = point_3d_kp_all_best[kk]
        if len(point_3d) >0 and len(RT_ceres)>3:
            #print(point_3d.reshape(4)[0:3])

            point_3d_kp_all_best[kk] = triangulate_nview_with_ceres(K_ceres, RT_ceres,point_3d.reshape(4)[0:3],point_2d)
        else:
            point_3d_kp_all_best[kk] = []
    #print(point_3d_kp_all_best)
    camera,cameras = count_inliers(all_bb,point_3d_kp_all_best,K_all,RT_all,kp_all,cam_index)
    return len(np.unique(camera)),cameras,point_3d_kp_all_best,reproject_error_kp

def check_nviews_nonlinear_1(RT_all,K_all,all_bb,cam_index,kp_all,point_3d_kp_all):
    camera,cameras = count_inliers_kp(all_bb,point_3d_kp_all,K_all,RT_all,kp_all,cam_index)
    num_added =  np.zeros(len(point_3d_kp_all))
    reproject_error_kp = 0
    point_3d_kp_eachview = []
    point_3d_kp_all_best = point_3d_kp_all
    
    correspondence = []
    cam = []
    RT_ceres = []
    K_ceres = []
    point_2d = []
    counter = 0
    for kk,loop in enumerate(point_3d_kp_all_best):
        corr_len = len(correspondence)
        for l,cams_l in enumerate(cameras):
            if kp_all[cams_l][kk,2]<50 or kk == 8 or kk == 9:# or kk > 3:
                continue
            K_ceres.append(K_all[cams_l])
            RT_ceres.append(RT_all[cams_l])
            point_2d = np.concatenate((point_2d,kp_all[cams_l][kk,0:2]))
            correspondence.append(kk)
            cam.append(cam_index[cams_l])
    #print(cam)
    keypoints = np.loadtxt('/home/dinesh/CarCrash/codes/CollisionRecon/keypoints/car_cad/car6_kp.txt', dtype='f', delimiter=',')
    #scale,RT_transform = scale_transformation(point_3d_kp_all,keypoints)
    scale = 1
    RT_transform = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    #print(RT_transform)
    #np.savez('1.npy', K_ceres, RT_ceres,keypoints.reshape(len(keypoints)*3),point_2d,correspondence,scale,RT_transform[0:3],)
    if len(K_ceres) >3:
        try:
            scale,RT_transform = car_fit_nview_with_ceres(K_ceres, RT_ceres,keypoints.reshape(len(keypoints)*3),point_2d,correspondence,scale,RT_transform[0:3])
        except:
            scale = scale
    #print(RT_transform)
    point_3d_car_kp_all_best =[]
    for kk,loop in enumerate(keypoints):
        keypnt = np.transpose(np.dot(RT_transform,np.transpose(np.append(keypoints[kk]*scale,1))))
        #print(np.append(keypoints[kk]*scale,1))
        #print(keypnt)
        if kk == 8 or kk == 9:
            point_3d_car_kp_all_best.append([])
        else:
            point_3d_car_kp_all_best.append(np.reshape(keypnt,[4,1]))
    
    #print(point_3d_kp_all_best)
    camera,cameras = count_inliers_kp(all_bb,point_3d_car_kp_all_best,K_all,RT_all,kp_all,cam_index)
    return len(np.unique(camera)),cameras,point_3d_car_kp_all_best,reproject_error_kp,scale,RT_transform


def ransac_nview(all_bb,kp_all,cam_index,RT_all,K_all):
    ransac_error = 100000
    ransac_num = 0
    bb_inside_final = []
    point_3d_kp_all_final = []
    scale_final = []
    car_RT_final  = []
    #print(cam_index)1
    for loop in range(300):
        bb_index,bb_compare_index = random.sample(range(1,len(cam_index)),2)
        #bb_index = 0
        #bb_compare_index = 3
        RT_1 = np.append(RT_all[bb_index],[0,0,0,1]).reshape(4,4)
        RT_2 = np.append(RT_all[bb_compare_index],[0,0,0,1]).reshape(4,4)
        RT_12 = np.dot(np.linalg.inv(RT_1),RT_2)
        P_1 = np.dot(K_all[bb_index],  RT_all[bb_index]) 
        P_2 = np.dot(K_all[bb_compare_index],  RT_all[bb_compare_index])
        if np.abs(np.mean(RT_12[0:3,3])) < 0.1:
            continue
        #print(np.mean(RT_12[1:3,3]))

        kp_1 =  kp_all[bb_index]
        kp_2 = kp_all[bb_compare_index]
        point_3d_kp_all = triangulate_visible_keypoints(P_1,P_2,kp_1,kp_2)
        camera,cameras = count_inliers_kp(all_bb,point_3d_kp_all,K_all,RT_all,kp_all,cam_index)
        if len(cameras) > 3:
            continue
        scale = 1
        RT_car = np.identity(4)
        #num_inside,bb_inside,point_3d_kp_all_loop,reproject_error = check_nviews_ransac(RT_all,K_all,all_bb,cam_index,kp_all,point_3d_kp_all)
        #num_inside,bb_inside,point_3d_kp_all_loop,reproject_error = check_nviews_nonlinear(RT_all,K_all,all_bb,cam_index,kp_all,point_3d_kp_all)
        #num_inside,bb_inside,point_3d_kp_all_loop,reproject_error,scale,RT_car = check_nviews_nonlinear_1(RT_all,K_all,all_bb,cam_index,kp_all,point_3d_kp_all)
        num_inside,bb_inside,point_3d_kp_all_loop,reproject_error,scale,RT_car = check_nviews_nonlinear_1(RT_all,K_all,all_bb,cam_index,kp_all,point_3d_kp_all)
        if  ransac_num < num_inside:#ransac_error: 
            #num_inside_new,bb_inside,point_3d_kp_all_loop,reproject_error,scale,RT_car = check_nviews_nonlinear_1(RT_all,K_all,all_bb,cam_index,kp_all,point_3d_kp_all_loop)
            ransac_num = num_inside
            scale_final = scale
            car_RT_final = RT_car

            #print(point_3d_kp_all_loop_ransac)
            #print(point_3d_kp_all_loop)
            point_3d_kp_all_final = point_3d_kp_all_loop# point_3d_kp_all_loop
            cams = []
            for a,b in enumerate(bb_inside):
                cams.append(cam_index[b])
            print(loop,num_inside,bb_inside,cams)
            bb_inside_final = bb_inside

    return point_3d_kp_all_final,bb_inside_final,scale_final,car_RT_final

def reconstruct_keypoints(all_bb,kp_all,cam_index,RT_all,K_all):
    bb_counter = 0
    Car_3d = []
    correspondence_all = []
    scale_all = []
    car_rt_all = []
    while len(cam_index)>3:
        point_3d_kp,bb_final,scale_final,car_RT_final = ransac_nview(all_bb,kp_all,cam_index,RT_all,K_all)
        #print(point_3d_kp)
        #print(bb_final)
        if len(bb_final) < 5:
            break
        Car_3d.append(point_3d_kp)
        correspondence_all.append(bb_final)
        scale_all.append(scale_final)
        car_rt_all.append(car_RT_final)
        # find the cars across views
        all_bb,kp_all,cam_index,RT_all,K_all,bb_counter = prun_bb(all_bb,kp_all,cam_index,RT_all,K_all,bb_counter,bb_final,point_3d_kp)
        #break
    return Car_3d,correspondence_all,scale_all,car_rt_all

def image_carfit(time,RT,RS, RT_index,K, K_index,diff,diff_ext):
            synched_images = - diff + time - diff_ext
            synched_images_ext = - diff + time #+ diff_ext
        
            ## read keypoints
            data = Read_keypoints(synched_images,Folder,leading_zeros)
            if Rollingshutter == True:
                all_bb,kp_all,cam_index,tracklet,RT_all,RS_all,K_all = time_instance_data_RS(data,synched_images_ext,RT,RS,RT_index,K,K_index)
            else:
                all_bb,kp_all,cam_index,tracklet,RT_all,K_all = time_instance_data(data,synched_images_ext,RT,RT_index,K,K_index)
            
            #if Rollingshutter == True:
            #    check_camera_params_RS(K_all,RT_all,RS_all,all_bb,time,Folder,synched_images,cam_index,leading_zeros)
            #else:
            #    check_camera_params(K_all,RT_all,all_bb,time,Folder,synched_images,cam_index,leading_zeros)
            #check_camera_params(K_all,RT_all,all_bb,time,Folder,synched_images,cam_index,leading_zeros)
            #dad
            ### Ransac two view triangulation
            Car_3d,correspondence_all,scale_all,car_rt_all = reconstruct_keypoints(all_bb,kp_all,cam_index,RT_all,K_all)        
            
            ## save data for visualization
            save_images(Car_3d,correspondence_all,cam_index,all_bb,K_all,RT_all,time,Folder,synched_images)
            
            ##
            #print(Car_3d)
            data_for_vis(Folder,time,Car_3d,scale_all,car_rt_all)


