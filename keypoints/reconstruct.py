#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:18:16 2017

@author: dinesh
"""
import numpy as np
from numpy.linalg import norm
from numpy import array,mat,sin,cos,dot,eye
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from random import randint
import random
from icp.icp import icp
Folder = '/home/dinesh/CarCrash/data/Fifth/'
K = [[1404.439337,0.0,987.855283],[0.0,1436.466596,539.339041],[0.0,0.0,1.0]]

def read_extrinsics(filename):
    with open(filename) as f:
        lines = f.readlines()
    index = []
    RT = []
    for line in lines:
        index.append(line.split(' ')[0])
        T = np.array(line.split(' ')[4:7]).astype(np.float).reshape(3,1)
        R,jacobian = cv2.Rodrigues(np.array(line.split(' ')[1:4]).astype(np.float))
        #T = -(np.dot(R.T,T))
        RT.append(np.concatenate((R, T), axis=1))
    return RT, index

def read_intrinsics(filename):
    with open(filename) as f:
        lines = f.readlines()
    index = []
    K = []
    for line in lines:
        index.append(line.split(' ')[0])
        Int = np.array(line.split(' ')[5:10]).astype(np.float)
        K_computed = [[Int[0],0.0,Int[3]],[0.0,Int[1],Int[4]],[0.0,0.0,1.0]]
        #print(Int)
        K.append(K_computed)
    return K, index

def Read_keypoints(synched_images):
    data = []
    for cam,num in enumerate(synched_images):
        data.append([])
        filename = Folder + str(cam) + '/keypoints_txt/' + str(num).zfill(5) + '.png.txt'
        try:
            with open(filename) as f:
                lines = f.readlines()
            for line in lines:
                data[cam].append(line.split('\n')[0].split(','))
        except:
            data[cam].append([])
    return data

def loop(synched_images,RT_1,bb,corr):
    for cam_comp,num_comp in enumerate(synched_images):
        if len(data[cam_comp]) > 1:
            for index_comp,keypoints_comp in enumerate(data[cam_comp]):
                bb_compare = np.array([keypoints_comp[1],keypoints_comp[2],keypoints_comp[3],keypoints_comp[4]]).astype(np.int)
                Index_num_compare = np.where(np.array(RT_index[cam_comp]).astype(np.int) == num_comp)
                if len(Index_num_compare[0]) == 1 and cam != cam_comp:
                    RT_compare = RT[cam_comp][int(Index_num_compare[0])]
                    center_bb = np.array([[bb_compare[0]+bb_compare[2]/2],[bb_compare[1]+bb_compare[3]/2]])
                    center_bb_compare = np.array([[bb_compare[0]+bb_compare[2]/2],[bb_compare[1]+bb_compare[3]/2]])
                    P_1 = np.dot(K,  RT_1) 
                    P_2 = np.dot(K,  RT_compare)
                    #print(RT_1)
                    point_4d_hom = cv2.triangulatePoints(P_1,P_2,center_bb,center_bb_compare)
                    #print(point_4d_hom)
                    #print('check')
                    point_4d_hom = point_4d_hom.astype(np.float)
                    point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
                    #print(point_4d_hom)
                    #point_4d = [[-3.2419],[0.0352],[1.0953],[1]];
                    #print(point_4d)
                    #print(RT_1)
                    point_2d_1 = np.dot(P_1,point_4d)
                    point_2d_1 = point_2d_1/np.tile(point_2d_1[-1, :], (3, 1))
                    #print(point_2d_1)

                    point_2d_2 = np.dot(P_2,point_4d)
                    point_2d_2 = point_2d_1/np.tile(point_2d_1[-1, :], (3, 1))
                    
                    if np.sqrt(np.mean((point_2d_1[0:-1] - center_bb)**2)) + np.sqrt(np.mean((point_2d_1[0:-1] - center_bb)**2)) < 1000:
                        print(cam_comp,int(Index_num_compare[0]),bb_compare)
                        print(point_2d_1,center_bb)
                    #if cam_comp == 0:
                    #    print(cam_comp)
                    #    print(int(Index_num_compare[0]))
                    #    print(RT_compare)
                    #print(np.sqrt(np.mean((point_2d_1[0:-1] - center_bb)**2)) + np.sqrt(np.mean((point_2d_2[0:-1] - center_bb_compare)**2)))
                    #print(np.sqrt(np.mean((point_2d_2[0:-1] - center_bb_compare)**2)))
                    #print('done')
                    #point_4d = point_4d[:3, :].T
                else:
                    continue
        else:
            corr.append([])

def correspondense(synched_images,data, RT, RT_index):
    corr = []
    for cam,num in enumerate(synched_images):
        Index_num = np.where(np.array(RT_index[cam]).astype(np.int) == num)
        if len(Index_num[0]) == 1:
            RT_1 = RT[cam][int(Index_num[0])]
            if len(data[cam]) > 1: #try:
                for index,keypoints in enumerate(data[cam]):
                    bb = np.array([keypoints[1],keypoints[2],keypoints[3],keypoints[4]]).astype(np.int)
                    print(cam,num,bb)
                    loop(synched_images,RT_1,bb,corr)
                    print('done')
            else:
                corr.append([])
        else:
            corr.append([])
    return corr

def ransac_correspondense(synched_images,data, RT, RT_index):
    corr = []
    ## find all boundix boxes
    all_bb = []
    cam_index = []
    RT_all = []
    for cam,num in enumerate(synched_images):
        Index_num = np.where(np.array(RT_index[cam]).astype(np.int) == num)
        if len(Index_num[0]) == 1:
                RT_cam = RT[cam][int(Index_num[0])]
                if len(data[cam]) > 1: #try:
                    for index,keypoints in enumerate(data[cam]):
                        all_bb.append(np.array([keypoints[1],keypoints[2],keypoints[3],keypoints[4]]).astype(np.int))
                        cam_index.append(cam)
                        RT_all.append(RT_cam)


    
    for cam,num in enumerate(synched_images):
        Index_num = np.where(np.array(RT_index[cam]).astype(np.int) == num)
        if len(Index_num[0]) == 1:
            RT_1 = RT[cam][int(Index_num[0])]
            if len(data[cam]) > 1: #try:
                for index,keypoints in enumerate(data[cam]):
                    bb = np.array([keypoints[1],keypoints[2],keypoints[3],keypoints[4]]).astype(np.int)
                    print(cam,num,bb)
                    loop(synched_images,RT_1,bb,corr)
                    print('done')
            else:
                corr.append([])
        else:
            corr.append([])
    return corr


def Draw_camera(synched_images,data, RT, RT_index):
    for cam,num in enumerate(synched_images):
        Index_num = np.where(np.array(RT_index[cam]).astype(np.int) == num)
        if len(Index_num[0]) == 1:
            RT_1 = RT[cam][int(Index_num[0])]
            print(RT_1)
        else:
            continue

def distance(point_1,point_2):
    return np.sqrt(np.mean((point_1-point_2)**2))

def Reproject_error(RT_all,K_all,all_bb,cam_index,location_3d):
    final_error = 0
    cam_num = 0
    center_boundingbox_list = []
    for i,boundingbox in enumerate(all_bb):
        P = np.dot(K_all[i],  RT_all[i])
        point_2d = np.dot(P,location_3d)
        point_2d = point_2d/np.tile(point_2d[-1, :], (3, 1))
        center_boundingbox = np.array([[boundingbox[0]+boundingbox[2]/2],[boundingbox[1]+boundingbox[3]/2]])
        size = boundingbox[3]**2 + boundingbox[2]**2
        if point_2d[0] > boundingbox[0] and point_2d[0] < boundingbox[0] + boundingbox[2] and point_2d[1] > boundingbox[1] and point_2d[1] < boundingbox[1] + boundingbox[3]:
            final_error += np.sqrt(np.mean((point_2d[0:-1] - center_boundingbox)**2))/np.sqrt(size)
    
    if final_error == 0:
        for i,boundingbox in enumerate(all_bb):
            P = np.dot(K_all[i],  RT_all[i])
            point_2d = np.dot(P,location_3d)
            point_2d = point_2d/np.tile(point_2d[-1, :], (3, 1))
            center_boundingbox = np.array([[boundingbox[0]+boundingbox[2]/2],[boundingbox[1]+boundingbox[3]/2]])
            center_boundingbox_list.append(center_boundingbox)
            if cam_num != cam_index[i]:
                #print(np.min(np.mean((center_boundingbox_list - point_2d[0:-1])**2)))
                final_error += np.sqrt(np.min(np.mean((center_boundingbox_list - point_2d[0:-1])**2)))
                cam_num = cam_index[i]
                center_boundingbox_list = []
    return final_error


def Reproject_error_1view(P_1,Point_1,location_3d):
        point_2d_1 = np.dot(P_1,location_3d)
        point_2d_1 = point_2d_1/np.tile(point_2d_1[-1, :], (3, 1))
        final_error = distance(point_2d_1[0:-1],Point_1)
        return final_error
def Reproject_error_2view(P_1,P_2,Point_1,Point_2,location_3d):
        final_error = Reproject_error_1view(P_1,Point_1,location_3d) + Reproject_error_1view(P_2,Point_2,location_3d)
        return final_error


def Reproject_error_nview(RT_all,K_all,all_bb,kp_all,kp_index,cam_index,location_3d):
    final_error = 0
    cam_num = 0
    center_boundingbox_list = []
    max_distance = 10000
    final_error_sample = 0
    count = 0
    for i,boundingbox in enumerate(all_bb):
        P = np.dot(K_all[i],  RT_all[i])
        point_2d = np.dot(P,location_3d)
        point_2d = point_2d/np.tile(point_2d[-1, :], (3, 1))
        point_kp = kp_all[i][kp_index].reshape(3,1).astype(np.float)
        if distance(point_kp,point_2d) < max_distance:
            final_error_sample = distance(point_kp,point_2d)
            max_distance = distance(point_kp,point_2d)
            #print(final_error_sample)

        if cam_num != cam_index[i]:
            #print('sa')
            if final_error_sample < 1000:
                final_error += final_error_sample
                count = count + 1
            final_error_sample = 0
            max_distance = 10000
            cam_num = cam_index[i]
    #print('done')
    if count == 0:
        count = 1
        final_error = 10000 
    return final_error/count

def check_nviews(RT_all,K_all,all_bb,cam_index,point_3d_kp_all):
    count_views = 0
    cam_num = 0
    for j,location_3d in enumerate(point_3d_kp_all):
        count = 0
        num_of_point = 0
        for i,boundingbox in enumerate(all_bb):
            P = np.dot(K_all[i],  RT_all[i])
            point_2d = np.dot(P,location_3d)
            point_2d = point_2d/np.tile(point_2d[-1, :], (3, 1))
            if point_2d[0] > boundingbox[0] and point_2d[0] < boundingbox[0] + boundingbox[2] and point_2d[1] > boundingbox[1] and point_2d[1] < boundingbox[1] + boundingbox[3]:
                num_of_point = num_of_point + 1
            if cam_num != cam_index[i]:
                cam_num = cam_index[i]
                if num_of_point>0:
                    count = count + 1
                    num_of_point = 0
        if count > len(point_3d_kp_all)/3 and count > 1:
            count_views = count_views + 1
    return count_views


   
def extrinsics(Folder):
    RT = []
    RT_index = []
    for cam in range(21):
        try:
            RT_cam, index_cam = read_extrinsics(Folder +  '/vHCamPose_RSCayley_' + str(cam) + '.txt')
        except:
            RT_cam = []
            index_cam = []#read_extrinsics('/home/dinesh/CarCrash/data/Fifth/avCamPose_RSCayley_'+ str(cam) + '.txt')
        RT.append(RT_cam)#[])
        RT_index.append(index_cam)#[])
        print('reading extrinsics of ' , cam)
    return RT,RT_index

def intrinsics(Folder):
    K = []
    K_index = []
    for cam in range(21):
        try:
            K_cam, index_K_cam = read_intrinsics(Folder + '/vHIntrinsic_'+ str(cam) + '.txt')
            #print(cam)
        except:
            K_cam = []
            index_K_cam = []#read_func('/home/dinesh/CarCrash/data/Fifth/avCamPose_RSCayley_'+ str(cam) + '.txt')
        K.append(K_cam)#[])
        K_index.append(index_K_cam)#[])
        print('reading intrinsics of ' , cam)
    return K,K_index
    
    
def cameraSync(Folder):
    with open(Folder + '/InitSync.txt') as f:
        lines = f.readlines()
    index = []
    diff = []
    for line in lines:
        index.append(line.split('\t')[0])
        diff.append(line.split('\t')[1].split('\r')[0])
    
    diff = np.array(diff).astype(int)
    return diff


def time_instance_data(synched_images,RT,RT_index,K,K_index):
    all_bb = []
    cam_index = []
    RT_all = []
    kp_all = []
    K_all = []
    for cam,num in enumerate(synched_images):
        Index_num = np.where(np.array(RT_index[cam]).astype(np.int) == num)
        if len(Index_num[0]) == 1:
                RT_cam = RT[cam][int(Index_num[0])]
                K_cam = K[cam][int(Index_num[0])]
                if len(data[cam]) > 1: #try:
                    for index,keypoints in enumerate(data[cam]):
                        bb =np.array([keypoints[1],keypoints[2],keypoints[3],keypoints[4]]).astype(np.int)
                        
                        points_array = np.array(keypoints[5:])
                        points_arranged = points_array.reshape(14,3)
                        kp = np.round(points_arranged.astype(np.float)).astype(np.int)
                        kp[:,0] = bb[0] + kp[:,0]*(bb[2]/64)
                        kp[:,1] = bb[1] + kp[:,1]*(bb[3]/64)
                        kp[:,2] = np.round(points_arranged[:,2].astype(np.float)*100).astype(np.int)
                        #print(np.sqrt(bb[2]**2+bb[3]**2))
                        if np.sqrt(bb[2]**2+bb[3]**2) > 100 and np.sqrt(bb[2]**2+bb[3]**2) < 1000 and cam != 6 and cam != 8: 
                            all_bb.append(bb)
                            kp_all.append(kp)
                            cam_index.append(cam)
                            RT_all.append(RT_cam)
                            K_all.append(K_cam)
    return all_bb,kp_all,cam_index,RT_all,K_all

def triangulate(P_1,P_2,keypoint_1,keypoint_2):
    point_3d_kp_hom = cv2.triangulatePoints(P_1,P_2,keypoint_1,keypoint_2)
    point_3d_kp_hom = point_3d_kp_hom.astype(np.float)
    point_3d_kp = point_3d_kp_hom / np.tile(point_3d_kp_hom[-1, :], (4, 1))
    return point_3d_kp


def ransac_nview(all_bb,kp_all,cam_index,RT_all,K_all):
    ransac_error = 100000
    ransac_num = 0
    location_3d = [0,0,0,0]
    for loop in range(1000):
        bb_index,bb_compare_index =random.sample(range(1,len(cam_index)),2)
        if cam_index[bb_index] == cam_index[bb_compare_index]:
            continue
        bb = all_bb[bb_index]
        bb_compare = all_bb[bb_compare_index]#random.sample(all_bb,2)
        center_bb = np.array([[bb[0]+bb[2]/2],[bb[1]+bb[3]/2]])
        center_bb_compare = np.array([[bb_compare[0]+bb_compare[2]/2],[bb_compare[1]+bb_compare[3]/2]])
        P_1 = np.dot(K_all[bb_index],  RT_all[bb_index]) 
        P_2 = np.dot(K_all[bb_compare_index],  RT_all[bb_compare_index])
        kp_1 =  kp_all[bb_index]
        kp_2 = kp_all[bb_compare_index]
        reproject_error_kp = 0
        point_3d_kp_all =[]
        point_3d_kp_all_loop = []
        count = 0
        point_3d_kp_sum = [0]
        for kp_i,kp_v in enumerate(kp_1):
            if kp_1[kp_i,2] >10 and kp_2[kp_i,2] >10:
                keypoint_1 = kp_1[kp_i,0:2].reshape(2,1).astype(np.float)
                keypoint_2 = kp_2[kp_i,0:2].reshape(2,1).astype(np.float)
                point_3d_kp = triangulate(P_1,P_2,keypoint_1,keypoint_2)
                point_3d_kp_all.append(point_3d_kp)
                point_3d_kp_all_loop.append(point_3d_kp)
                point_3d_kp_sum += point_3d_kp
                reproject_error_kp += Reproject_error_2view(P_1,P_2,keypoint_1,keypoint_2,point_3d_kp)
                #reproject_error_kp += Reproject_error_nview(RT_all,K_all,all_bb,kp_all,kp_i,cam_index,point_3d_kp)
                count = count + 1
            else:
                point_3d_kp_all_loop.append([])
        if count != 0:
            reproject_error_kp = reproject_error_kp/count
        else:
            reproject_error_kp = 100000
        
        point_3d_hom = cv2.triangulatePoints(P_1,P_2,center_bb,center_bb_compare)
        point_3d_hom = point_3d_hom.astype(np.float)
        point_3d = point_3d_hom / np.tile(point_3d_hom[-1, :], (4, 1))
        num_inside = check_nviews(RT_all,K_all,all_bb,cam_index,point_3d_kp_all)
        #reproject_error = Reproject_error_2view(P_1,P_2,center_bb,center_bb_compare,point_3d)
        reproject_error = reproject_error_kp# + reproject_error_kp
        #reproject_error = Reproject_error(RT_all,K_all,all_bb,cam_index,point_3d)
        
        if num_inside > ransac_num:
            point_3d = point_3d_kp_sum/count
            #print(point_3d_kp_all_final)
            point_3d_kp_all_final = point_3d_kp_all_loop
            location_3d = point_3d
            ransac_num = num_inside
      #    reproject_error
#        if reproject_error < ransac_error:
##            print(reproject_error)
#            point_3d = point_3d_kp_sum/count
#            location_3d = point_3d
#            ransac_error = reproject_error
    #print(location_3d)
    return point_3d_kp_all_final,location_3d

def find_matches(all_bb,kp_all,cam_index,RT_all,K_all,bb_counter):
    all_bb_new = []
    cam_index_new = []
    RT_all_new = []
    K_all_new =[]
    kp_all_new = []
    c = (random.randint(1,255),random.randint(1,255),random.randint(1,255))
    
    
    for i,boundingbox in enumerate(all_bb):
        P_1 = np.dot(K_all[i],  RT_all[i])
        point_2d = np.dot(P_1,location_3d)
        point_2d = point_2d/np.tile(point_2d[-1, :], (3, 1))
        #center_boundingbox = np.array([[boundingbox[0]+boundingbox[2]/2],[boundingbox[1]+boundingbox[3]/2]])
        #img1 = cv2.imread(Folder + str(cam_index[i]) + '/' + str(synched_images[cam_index[i]]).zfill(5) + '.png' )
        #cv2.rectangle(img1,(int(boundingbox[0]),int(boundingbox[1])),(int(boundingbox[0] + boundingbox[2]),int(boundingbox[1]+boundingbox[3])),c,3)
        #cv2.imwrite(str(synched_images[cam_index[i]]).zfill(5)  + '_check.png',img1)
        if point_2d[0] > boundingbox[0] and point_2d[0] < boundingbox[0] + boundingbox[2] and point_2d[1] > boundingbox[1] and point_2d[1] < boundingbox[1] + boundingbox[3]:
            # print()
            # print(np.sqrt(np.mean((point_2d[0:-1] - center_boundingbox)**2)))
            img = cv2.imread(Folder + str(cam_index[i]) + '/' + str(synched_images[cam_index[i]]).zfill(5) + '.png' )
            cv2.rectangle(img,(int(boundingbox[0]),int(boundingbox[1])),(int(boundingbox[0] + boundingbox[2]),int(boundingbox[1]+boundingbox[3])),c,3)
            cv2.circle(img,tuple(point_2d[0:2]),3,(255,0,128),5)
            cv2.imwrite(str(bb_counter) + '.png',img)
            bb_counter += 1
        else:
            all_bb_new.append(boundingbox)
            cam_index_new.append(cam_index[i])
            RT_all_new.append(RT_all[i])
            K_all_new.append(K_all[i])
            kp_all_new.append(kp_all[i])
    #print(len(all_bb))
    return all_bb_new,kp_all_new,cam_index_new,RT_all_new,K_all_new,bb_counter

# Read extrinsicsd
RT, RT_index = extrinsics(Folder)
# Read intrinsics\
K, K_index = intrinsics(Folder)
# Read cameraSync
diff = cameraSync(Folder)

synched_images = - diff + 100


# read keypoints

data = Read_keypoints(synched_images)

# find matches
#matches = correspondense(synched_images,data, RT,RT_index)

# ransac
#matches = ransac_correspondense(synched_images,data, RT,RT_index)

## find all boundix boxes
all_bb,kp_all,cam_index,RT_all,K_all = time_instance_data(synched_images,RT,RT_index,K,K_index)

## Ransac two view triangulation
bb_counter = 0
Car_3d = []
while len(all_bb) >6:

    point_3d_kp,location_3d = ransac_nview(all_bb,kp_all,cam_index,RT_all,K_all)
    #print(point_3d_kp)
    #print(location_3d)
    Car_3d.append(point_3d_kp)
    # find the cars across views
    all_bb,kp_all,cam_index,RT_all,K_all,bb_counter = find_matches(all_bb,kp_all,cam_index,RT_all,K_all,bb_counter)
    
    print(bb_counter)
    #aSDASD


keypoints = np.loadtxt('/home/dinesh/CarCrash/codes/CollisionRecon/keypoints/car_cad/car1_kp.txt', dtype='f', delimiter=',')
for loop,car_points_3d in enumerate(Car_3d):
    points_mesh = []
    points_ours = []
    for ind,points in enumerate(car_points_3d):
        if points == [] or ind == 8 or ind ==9:
            ind = ind
        else:
            points_ours.append(points)
            points_mesh.append(keypoints[ind])
            #scale = 
    # compute scale
    scale_sum = 0
    coun = 0
    scale =[0] * 100
    for indices,points_3d in enumerate(points_ours):
        for indices_2,points_3d_2 in enumerate(points_ours):
            if indices_2 > indices and np.abs(np.mean(points_3d - points_3d_2))/np.abs(np.mean(points_mesh[indices] - points_mesh[indices_2])) < 5:
                scale[coun] = np.abs(np.mean(points_3d - points_3d_2))/np.abs(np.mean(points_mesh[indices] - points_mesh[indices_2]))
                coun += 1
    
    # compute transformation
    scale_all = scale[0:coun]
    ransac_max = 10000
    for i,scale_check in enumerate(scale_all):
        points_ours_icp = np.array([[0,0,0]] * len(points_ours)).astype(np.float)
        points_mesh_icp = np.array([[0,0,0]] * len(points_ours)).astype(np.float)
        for indices,points_3d in enumerate(points_ours):
            points_ours_icp[indices] = points_ours[indices][0:3].T[0]
            points_mesh_icp[indices] = points_mesh[indices][0:3].T[0]*scale_check
        try:
            RT_transform,distance = icp(points_ours_icp, points_mesh_icp, None, 20, 0.001)
            if ransac_max > np.mean(distance):
                print(ransac_max)
                ransac_max = np.mean(distance)
                RT_transform_best = RT_transform
                scale_best = scale_check

        except:
            print('icp failed')#scale = print(points_3d)
        
    print(points_ours_icp[0])
    print(np.dot(np.linalg.inv(RT_transform_best),np.transpose(np.append(points_mesh_icp[0],1))))
    