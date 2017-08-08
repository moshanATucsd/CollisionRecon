#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 17:11:44 2017

@author: dinesh
"""
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from geometry import *
import random
import sys
sys.path.append("/home/dinesh/CarCrash/codes/CollisionRecon/keypoints/python_ceres/")
from python_ceres.ba import *

mutable_object ={}

def on_press(event):
    print('you pressed', event.button, event.xdata, event.ydata)
    plt.close()
    mutable_object['key'] =  [event.xdata, event.ydata]  
    return([event.xdata, event.ydata])

def keypoint_press(filename):
    keypoints = []
    for i in range(2):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = Image.open(filename)
        arr = np.asarray(im)
        plt_image=plt.imshow(arr)
        cid = fig.canvas.mpl_connect('button_press_event', on_press)
        plt.show()
        keypoints.append(mutable_object['key'])
    return(keypoints)

def check_camera_params(K_all,RT_all,all_bb,time,Folder,synched_images,cam_index,leading_zeros):
    point_3d_kp_all = []
    
    keypoints_1 = keypoint_press(Folder + str(0) + '/' + str(synched_images[0]).zfill(leading_zeros) + '.png')
    keypoints_2 = keypoint_press(Folder + str(1) + '/' + str(synched_images[1]).zfill(leading_zeros) + '.png')
    #keypoints_1 =[[1614.30032138, 768.624101051]]
    #keypoints_2 =[[1041.02775217, 609.302692237]]#print(point_2d_2)
    index_0 =np.where(np.array(cam_index) == 0)[0][0]
    index_1 =np.where(np.array(cam_index) == 1)[0][0]
    P_1 = ProjectionMatrix(K_all[index_0],  RT_all[index_0]) 
    P_2 = ProjectionMatrix(K_all[index_1],  RT_all[index_1])
    for li,kp_1 in enumerate(keypoints_1):
        keypoint_1 = np.array(kp_1)
        keypoint_2 = np.array(keypoints_2[li]);
        point_3d = triangulate(P_1,P_2,keypoint_1,keypoint_2)
        point_2d = np.append(keypoint_1,keypoint_2)
        point_3d_kp_all.append(point_3d)
        #triangulate_with_ceres(K_all[index_0],K_all[index_1], RT_all[index_0],RT_all[index_1],point_3d[0:3].reshape(3),point_2d)
#        triangulate_nview_with_ceres(K_all, RT_all,point_3d[0:3].reshape(3)*0,point_2d)

        #point_3d_kp_all.append(triangulate_with_ceres(K_all[index_0],K_all[index_1], RT_all[index_0],RT_all[index_1],point_3d[0:3].reshape(3),point_2d))
    #point_3d_kp_all.append([[1.1954653612647386],[ -0.9993841072343063],[ 6.7841787420247686],[1]])  
    
    loop = cam_index[0]
    c = []
    for kll in range(20):
        c.append((random.randint(1,255),random.randint(1,255),random.randint(1,255)))
    for i,boundingbox in enumerate(all_bb):
        if cam_index[i] != loop:
            cv2.imwrite(str(loop) + str(time) + '.png',img)
            loop = cam_index[i]

        P_1 = np.dot(K_all[i],  RT_all[i])
#        print(Folder + str(cam_index[i]) + '/' + str(synched_images[cam_index[i]]).zfill(5) + '.png')
        img = cv2.imread(Folder + str(cam_index[i]) + '/' + str(synched_images[cam_index[i]]).zfill(leading_zeros) + '.png' )
        print(Folder + str(cam_index[i]) + '/' + str(synched_images[cam_index[i]]).zfill(leading_zeros) + '.png')
        for sdas,point_3d in enumerate(point_3d_kp_all):
            if len(point_3d) > 2:
                    point = np.dot(P_1,point_3d)
                    point = point/np.tile(point[-1, :], (3, 1))
                    cv2.circle(img,tuple(point[0:2]),3,c[sdas],5)
    print(point_3d_kp_all)


def check_camera_params_RS(K_all,RT_all,RS_all,all_bb,time,Folder,synched_images,cam_index,leading_zeros):
    point_3d_kp_all = []
    
    #keypoints_1 = keypoint_press(Folder + str(0) + '/' + str(synched_images[0]).zfill(leading_zeros) + '.png')
    #keypoints_2 = keypoint_press(Folder + str(1) + '/' + str(synched_images[1]).zfill(leading_zeros) + '.png')
    keypoints_1 =[[1614.30032138, 768.624101051]]
    keypoints_2 =[[1041.02775217, 609.302692237]]
    #print(point_2d_2)
    index_0 =np.where(np.array(cam_index) == 0)[0][0]
    index_1 =np.where(np.array(cam_index) == 1)[0][0]
    for li,kp_1 in enumerate(keypoints_1):
        P_1 = ProjectionMatrix_RS(keypoints_1[li],K_all[index_0],  RT_all[index_0],RS_all[index_0]) 
        P_2 = ProjectionMatrix_RS(keypoints_2[li],K_all[index_1],  RT_all[index_1],RS_all[index_1])
        #print(RT_all[index_0])
        #print(P_2)
        keypoint_1 = np.array(kp_1)
        keypoint_2 = np.array(keypoints_2[li]);
        point_3d = triangulate(P_1,P_2,keypoint_1,keypoint_2)
        point_2d = np.append(keypoint_1,keypoint_2)
        point_3d_kp_all.append(point_3d)
        #triangulate_with_ceres(K_all[index_0],K_all[index_1], RT_all[index_0],RT_all[index_1],point_3d[0:3].reshape(3),point_2d)
        #triangulate_nview_with_ceres(K_all, RT_all,point_3d[0:3].reshape(3)*0,point_2d)
        #point_3d_kp_all.append(triangulate_with_ceres(K_all[index_0],K_all[index_1], RT_all[index_0],RT_all[index_1],point_3d[0:3].reshape(3),point_2d))
    point_3d_kp_all.append([[-2.309049],[ -0.292767],[ 3.877774],[1]])  
    
    loop = cam_index[0]
    c = []
    for kll in range(20):
        c.append((random.randint(1,255),random.randint(1,255),random.randint(1,255)))
    for i,boundingbox in enumerate(all_bb):
        if cam_index[i] != loop:
            cv2.imwrite(str(loop) + str(time) + '.png',img)
            loop = cam_index[i]

        P_1 = np.dot(K_all[i],  RT_all[i])
#        print(Folder + str(cam_index[i]) + '/' + str(synched_images[cam_index[i]]).zfill(5) + '.png')
        img = cv2.imread(Folder + str(cam_index[i]) + '/' + str(synched_images[cam_index[i]]).zfill(leading_zeros) + '.png' )
        print(Folder + str(cam_index[i]) + '/' + str(synched_images[cam_index[i]]).zfill(leading_zeros) + '.png')
        for sdas,point_3d in enumerate(point_3d_kp_all):
            if len(point_3d) > 2:
                    point = np.dot(P_1,point_3d)
                    point = point/np.tile(point[-1, :], (3, 1))
                    cv2.circle(img,tuple(point[0:2]),3,c[sdas],5)
    print(point_3d_kp_all)


def check_nonlinear(K_all,RT_all,all_bb,time,Folder,synched_images,cam_index):
    point_3d_kp_all = []
    
    RT_ceres = []
    K_ceres = []
    point_2d = []
    keypoints_1 =[[0],[0]]
    for a in range(20):
        keypoints_1_old = keypoints_1
        try:
            keypoints_1 = keypoint_press(Folder + str(a) + '/' + str(synched_images[a]).zfill(5) + '.png')
        except:
            continue
        print(keypoints_1,a)
        if keypoints_1_old[0][0] != keypoints_1[0][0]:
            cams_l = np.where(np.array(cam_index) == a)[0][0]
            RT_ceres.append(RT_all[cams_l])
            K_ceres.append(K_all[cams_l])
            #print(kp_all[cams_l][kk,0:2])
            print(keypoints_1[0])
            point_2d = np.concatenate((point_2d,keypoints_1[0]))
    if len(K_ceres) >2:
        point_3d_kp_all.append(triangulate_nview_with_ceres(K_ceres, RT_ceres,[0,0,0],point_2d))


    loop = cam_index[0]
    c = []
    for kll in range(20):
        c.append((random.randint(1,255),random.randint(1,255),random.randint(1,255)))
    for i,boundingbox in enumerate(all_bb):
        if cam_index[i] != loop:
            cv2.imwrite(str(loop) + str(time) + '.png',img)
            loop = cam_index[i]

        P_1 = np.dot(K_all[i],  RT_all[i])
#        print(Folder + str(cam_index[i]) + '/' + str(synched_images[cam_index[i]]).zfill(5) + '.png')
        img = cv2.imread(Folder + str(cam_index[i]) + '/' + str(synched_images[cam_index[i]]).zfill(5) + '.png' )
        print(Folder + str(cam_index[i]) + '/' + str(synched_images[cam_index[i]]).zfill(5) + '.png')
        for sdas,point_3d in enumerate(point_3d_kp_all):
            if len(point_3d) > 2:
                    point = np.dot(P_1,point_3d)
                    point = point/np.tile(point[-1, :], (3, 1))
                    cv2.circle(img,tuple(point[0:2]),3,c[sdas],5)
    print(point_3d_kp_all)
