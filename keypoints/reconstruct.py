#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:18:16 2017

@author: dinesh
"""
import numpy as np


Folder = '/home/dinesh/CarCrash/data/Fifth/'


def check_nviews_new(RT_all,K_all,all_bb,cam_index,kp_all,point_3d_kp_all):
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
            if np.mean(RT_12[0:3,3]) < 0.3:
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
#    keypoints = np.loadtxt('/home/dinesh/CarCrash/codes/CollisionRecon/keypoints/car_cad/car6_kp.txt', dtype='f', delimiter=',')
#    scale,RT_transform = scale_transformation(point_3d_kp_all,keypoints)
#    point_3d_car_kp_all_best =[]
#    for kk,loop in enumerate(keypoints):
#        keypnt = np.transpose(np.dot(RT_transform,np.transpose(np.append(keypoints[kk],1))))
#        if kk == 8 or kk == 9:
#            point_3d_car_kp_all_best.append([])
#        else:
#            point_3d_car_kp_all_best.append(np.reshape(keypnt,[4,1]))
        
    #print(keypoints)
    #print(point_3d_car_kp_all_best)
    camera,cameras = count_inliers(all_bb,point_3d_kp_all_best,K_all,RT_all,kp_all,cam_index)
    return len(np.unique(camera)),cameras,point_3d_kp_all_best,reproject_error_kp

def ransac_nview(all_bb,kp_all,cam_index,RT_all,K_all):
    ransac_error = 100000
    ransac_num = 0
    bb_inside_final = []
    point_3d_kp_all_final = []
    for loop in range(500):
        bb_index,bb_compare_index = random.sample(range(1,len(cam_index)),2)
        #bb_index = 0
        #bb_compare_index = 3
        RT_1 = np.append(RT_all[bb_index],[0,0,0,1]).reshape(4,4)
        RT_2 = np.append(RT_all[bb_compare_index],[0,0,0,1]).reshape(4,4)
        RT_12 = np.dot(np.linalg.inv(RT_1),RT_2)
        P_1 = np.dot(K_all[bb_index],  RT_all[bb_index]) 
        P_2 = np.dot(K_all[bb_compare_index],  RT_all[bb_compare_index])
        if np.mean(RT_12[0:3,3]) < 1:
            continue
        #print(np.mean(RT_12[1:3,3]))

        kp_1 =  kp_all[bb_index]
        kp_2 = kp_all[bb_compare_index]
        point_3d_kp_all = triangulate_visible_keypoints(P_1,P_2,kp_1,kp_2)

        num_inside,bb_inside,point_3d_kp_all_loop,reproject_error = check_nviews_new(RT_all,K_all,all_bb,cam_index,kp_all,point_3d_kp_all)
        
        #print(num_inside)
        if  ransac_num < num_inside:#ransac_error: 
            ransac_num = num_inside
            point_3d_kp_all_final = point_3d_kp_all_loop# point_3d_kp_all_loop
            print(loop,num_inside,bb_inside)
            bb_inside_final =bb_inside

    return point_3d_kp_all_final,bb_inside_final

        




def reconstruct_keypoints(all_bb,kp_all,cam_index,RT_all,K_all):
    bb_counter = 0
    Car_3d = []
    correspondence_all = []
    while 1:
        point_3d_kp,bb_final = ransac_nview(all_bb,kp_all,cam_index,RT_all,K_all)
        #print(point_3d_kp)

        #print(bb_final)
        if len(bb_final) < 7:
            break
        Car_3d.append(point_3d_kp)
        correspondence_all.append(bb_final)
        # find the cars across views
        all_bb,kp_all,cam_index,RT_all,K_all,bb_counter = prun_bb(all_bb,kp_all,cam_index,RT_all,K_all,bb_counter,bb_final,point_3d_kp)
        break
    return Car_3d,correspondence_all
            
            
RT, RT_index,K, K_index,diff,diff_ext = Read_data(Folder)
diff = cameraSync(Folder + '/InitSync_minh.txt')
diff_ext = cameraSync(Folder + '/InitSync.txt')
time = 650
while time < 651:
    time = time + 1
    synched_images = - diff + time 
    synched_images_ext = - diff + time + diff_ext
    ## read keypoints
    data = Read_keypoints(synched_images,Folder)
    all_bb,kp_all,cam_index,RT_all,K_all = time_instance_data(data,synched_images_ext,RT,RT_index,K,K_index)
    
    
    #check_camera_params(K_all,RT_all,all_bb,time)
    #sdad
    ### Ransac two view triangulation
    Car_3d,correspondence_all = reconstruct_keypoints(all_bb,kp_all,cam_index,RT_all,K_all)        
    
    ## save data for visualization
    save_images(Car_3d,correspondence_all,cam_index,all_bb,K_all,RT_all,time,Folder,synched_images)
    
    ## 
#    file = open(Folder + '/Car3D/' + str(time).zfill(5) +  '.txt','w') 
#    keypoints = np.loadtxt('/home/dinesh/CarCrash/codes/CollisionRecon/keypoints/car_cad/car6_kp.txt', dtype='f', delimiter=',')
#    for loop,car_points_3d in enumerate(Car_3d):
#        scale,RT_transform = scale_transformation(car_points_3d,keypoints)
#        write_car_transformations(file,scale,RT_transform)            
#    file.close()
