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

Folder = '/home/dinesh/CarCrash/data/Fifth/'
#Folder = '/home/dinesh/CarCrash/data/syn/'


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
        scale,RT_transform = car_fit_nview_with_ceres(K_ceres, RT_ceres,keypoints.reshape(len(keypoints)*3),point_2d,correspondence,scale,RT_transform[0:3])
    #print(RT_transform)
#    
    point_3d_car_kp_all_best =[]
    for kk,loop in enumerate(keypoints):
        keypnt = np.transpose(np.dot(RT_transform,np.transpose(np.append(keypoints[kk]*scale,1))))
        #print(np.append(keypoints[kk]*scale,1))
        #print(keypnt)
        if kk == 8 or kk == 9:
            point_3d_car_kp_all_best.append([])#It's showing "II Sem Fees*", is that an issue?
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
        if len(bb_final) < 4:
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





def reconstruct_keypoints_video(videodata,video_window):
    bb_counter = 0
    Car_3d = []
    correspondence_all = []
    scale_all = []
    car_rt_all = []
    #tracks = track_boundingbox(videodata)
    #print(int((len(videodata)-1)/2))
    #print(videodata[int((len(videodata)-1)/2)]['cam_index'])
    time = int((len(videodata)-1)/2)
    while len(videodata[time]['cam_index'])>3:
        point_3d_kp,bb_final,scale_final,car_RT_final = ransac_nview(videodata[time]['bb'],videodata[time]['kp'],videodata[time]['cam_index'],videodata[time]['RT'],videodata[time]['K'])
        
        #print(point_3d_kp)

        #print(bb_final)
        if len(bb_final) < 4:
            break
        Car_3d.append(point_3d_kp)
        correspondence_all.append(bb_final)
        scale_all.append(scale_final)
        car_rt_all.append(car_RT_final)
        # find the cars across views
        all_bb,kp_all,cam_index,RT_all,K_all,bb_counter = prun_bb(all_bb,kp_all,cam_index,RT_all,K_all,bb_counter,bb_final,point_3d_kp)
        #break
    return Car_3d,correspondence_all,scale_all,car_rt_all


def data_for_vis(Folder,time,Car_3d,scale_all,car_rt_all):
    file_rt = open(Folder + '/Car3D/' + str(time).zfill(5) +  '.txt','w') 
    keypoints = np.loadtxt('/home/dinesh/CarCrash/codes/CollisionRecon/keypoints/car_cad/car6_kp.txt', dtype='f', delimiter=',')
    
    for loop,car_points_3d in enumerate(Car_3d):
        for p_l,point_3d in enumerate(car_points_3d):
            if len(point_3d) == 0:
                continue
            file = open(Folder + '/Track3D/OptimizedRaw_Track_' + str(p_l+1000) +  '.txt','a')
            write_points_3d(file,point_3d,time,time)
            file.close()
        #file.write('\n')
        #scale,RT_transform = scale_transformation(car_points_3d,keypoints)
        #print(car_rt_all)
        write_car_transformations(file_rt,scale_all[loop],car_rt_all[loop])           
    file_rt.close()
    

def tracklet_data(tracklet_num,camera,time,kp_track,all_time,all_cam,camera_index):
    
    new_time = time
    flag = 0
    while(1):
        new_time = new_time - 1
        synched_images = - diff - diff_ext + new_time  
        data = Read_keypoints(synched_images,Folder,leading_zeros)
        for ind,kp in enumerate(data[camera]):
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

     
def video_carfit(time,RT,RS, RT_index,K, K_index,diff,diff_ext,video_window):
            synched_images = - diff + time - diff_ext
            synched_images_ext = - diff + time #+ diff_ext
        
            ## read keypoints
            data = Read_keypoints(synched_images,Folder,leading_zeros)
            all_bb,kp_all,cam_index,tracklet,RT_all,K_all = time_instance_data(data,synched_images_ext,RT,RT_index,K,K_index)
            
            Car_3d,correspondence_all,scale_all,car_rt_all = reconstruct_keypoints(all_bb,kp_all,cam_index,RT_all,K_all)       
            
            
            

            for ind,corr in enumerate(correspondence_all):
                kp_track = []
                all_cam = []
                all_time = []
                camera_index = []
                for a,val in enumerate(corr):
                    tracklet_data(tracklet[val],cam_index[val],time,kp_track,all_time,all_cam,camera_index)
            
                point_2d = []
                K_ceres = []
                RT_ceres = []
                correspondence = []
                car_time = []
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
                
                
                RT_transform = []
                keypoints = np.loadtxt('/home/dinesh/CarCrash/codes/CollisionRecon/keypoints/car_cad/car6_kp.txt', dtype='f', delimiter=',')
                scale = 1
                for a in range(max(car_time)):
                    RT_transform.append(np.identity(4))
                
                #np.savez('car_video', K_ceres, RT_ceres,keypoints.reshape(len(keypoints)*3),point_2d,correspondence,scale,RT_transform,car_time)
                scale,RT_transform = car_fit_video_nview_with_ceres(K_ceres, RT_ceres,keypoints.reshape(len(keypoints)*3),point_2d,correspondence,scale,RT_transform,car_time)
    

                
                
            
            

num_cams = 21
leading_zeros = 5
video_window = 11
Rollingshutter = False 
#RT,RS, RT_index,K, K_index,diff,diff_ext = Read_data(Folder,num_cams,Rollingshutter)
time = 630
while time < 631:
    time = time + 1
    #image_carfit(time,RT,RS, RT_index,K, K_index,diff,diff_ext)
    video_carfit(time, RT, RS, RT_index, K, K_index, diff, diff_ext, video_window)


    #print(len(kp_track))


            