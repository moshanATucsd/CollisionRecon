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

Folder = '/home/dinesh/CarCrash/data/Fifth/'
K = [[1404.439337,0.0,987.855283],[0.0,1436.466596,539.339041],[0.0,0.0,1.0]]

def read_func(filename):
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
        
## during triangulation
            #points_array = np.array(keypoints[5:])
            #kp = points_array.reshape(14,3)
            #kp = np.round(kp.astype(np.float)).astype(np.int)
            #kp[:,0] = bb[0] + kp[:,0]*(bb[2]/64)
            #kp[:,1] = bb[1] + kp[:,1]*(bb[3]/64)
            #kp[:,2] = np.round(points_arranged[:,2].astype(np.float)*100).astype(np.int)
            #print(bb)
  
    
# Read extrinsicsd
#RT = []
#RT_index = []
#for cam in range(21):
#    try:
#        RT_cam, index_cam = read_func('/home/dinesh/CarCrash/data/Fifth/vHCamPose_RSCayley_'+ str(cam) + '.txt')
#    except:
#        RT_cam = []
#        index_cam = []#read_func('/home/dinesh/CarCrash/data/Fifth/avCamPose_RSCayley_'+ str(cam) + '.txt')
#    RT.append(RT_cam)#[])
#    RT_index.append(index_cam)#[])
#    print(cam)




# Read cameraSync
with open('/home/dinesh/CarCrash/data/Fifth/InitSync.txt') as f:
        lines = f.readlines()

index = []
diff = []
for line in lines:
    index.append(line.split('\t')[0])
    diff.append(line.split('\t')[1].split('\r')[0])

diff = np.array(diff).astype(int)
synched_images = - diff 


# read keypoints

data = Read_keypoints(synched_images)

# find matches
#matches = correspondense(synched_images,data, RT,RT_index)

# ransac
#matches = ransac_correspondense(synched_images,data, RT,RT_index)

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


def Reproject_error(RT_all,all_bb,location_3d):
    final_error = 0
    for i,boundingbox in enumerate(all_bb):
        P = np.dot(K,  RT_all[i])
        point_2d = np.dot(P,location_3d)
        point_2d = point_2d/np.tile(point_2d[-1, :], (3, 1))
        center_boundingbox = np.array([[boundingbox[0]+boundingbox[2]/2],[boundingbox[1]+boundingbox[3]/2]])
        if point_2d[0] > boundingbox[0] and point_2d[0] < boundingbox[0] + boundingbox[2] and point_2d[1] > boundingbox[1] and point_2d[1] < boundingbox[1] + boundingbox[3]:
            final_error += np.sqrt(np.mean((point_2d[0:-1] - center_boundingbox)**2))
    return final_error

## Ransac two view triangulation

ransac_error = 10000
location_3d = [0,0,0]
for loop in range(10000):
    bb_index,bb_compare_index =random.sample(range(1,len(cam_index)),2)
    if cam_index[bb_index] == cam_index[bb_compare_index]:
        continue
    
    bb = all_bb[bb_index]
    bb_compare = all_bb[bb_compare_index]#random.sample(all_bb,2)
    center_bb = np.array([[bb[0]+bb[2]/2],[bb[1]+bb[3]/2]])
    center_bb_compare = np.array([[bb_compare[0]+bb_compare[2]/2],[bb_compare[1]+bb_compare[3]/2]])
    P_1 = np.dot(K,  RT_all[bb_index]) 
    P_2 = np.dot(K,  RT_all[bb_compare_index])

    point_4d_hom = cv2.triangulatePoints(P_1,P_2,center_bb,center_bb_compare)
    point_4d_hom = point_4d_hom.astype(np.float)
    point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))

    point_2d_1 = np.dot(P_1,point_4d)
    point_2d_1 = point_2d_1/np.tile(point_2d_1[-1, :], (3, 1))
    point_2d_2 = np.dot(P_2,point_4d)
    point_2d_2 = point_2d_2/np.tile(point_2d_2[-1, :], (3, 1))
    reproject_error = np.sqrt(np.mean((point_2d_1[0:-1] - center_bb)**2)) + np.sqrt(np.mean((point_2d_2[0:-1] - center_bb_compare)**2))
    if reproject_error < ransac_error:
        print(reproject_error)
        print(bb,cam_index[bb_index])
        print(bb_compare,cam_index[bb_compare_index])
        print(point_2d_1)

        location_3d = point_4d
        ransac_error = reproject_error
        print(Folder + str(cam_index[bb_index]) + '/' + str(synched_images[cam_index[bb_index]]).zfill(5) + '.png')
        img_1 = cv2.imread(Folder + str(cam_index[bb_index]) + '/' + str(synched_images[cam_index[bb_index]]).zfill(5) + '.png' )
        cv2.rectangle(img_1,(int(bb[0]),int(bb[1])),(int(bb[0] + bb[2]),int(bb[1]+bb[3])),(0,255,0))
        img_2 = cv2.imread(Folder + str(cam_index[bb_compare_index]) + '/' + str(synched_images[cam_index[bb_compare_index]]).zfill(5) + '.png' )
        cv2.rectangle(img_2,(int(bb_compare[0]),int(bb_compare[1])),(int(bb_compare[0] + bb_compare[2]),int(bb_compare[1]+bb_compare[3])),(0,255,0))
        #cv2.imwrite('1.png',img_1)
        #cv2.imwrite('2.png',img_2)

#Car_3d.append(location_3d)

# find the cars across views
print(location_3d)
for i,boundingbox in enumerate(all_bb):
    P_1 = np.dot(K,  RT_all[i])
    point_2d = np.dot(P_1,location_3d)
    point_2d = point_2d/np.tile(point_2d[-1, :], (3, 1))
    center_boundingbox = np.array([[boundingbox[0]+boundingbox[2]/2],[boundingbox[1]+boundingbox[3]/2]])
    if point_2d[0] > boundingbox[0] and point_2d[0] < boundingbox[0] + boundingbox[2] and point_2d[1] > boundingbox[1] and point_2d[1] < boundingbox[1] + boundingbox[3]:
        print(np.sqrt(np.mean((point_2d[0:-1] - center_boundingbox)**2)))
        print(cam_index[i])
        img = cv2.imread(Folder + str(cam_index[i]) + '/' + str(synched_images[cam_index[i]]).zfill(5) + '.png' )
        cv2.rectangle(img,(int(boundingbox[0]),int(boundingbox[1])),(int(boundingbox[0] + boundingbox[2]),int(boundingbox[1]+boundingbox[3])),(0,255,0))
        cv2.imwrite(str(i) + '.png',img)

    
