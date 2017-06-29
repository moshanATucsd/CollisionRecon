#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:18:16 2017

@author: dinesh
"""
import numpy as np
from numpy.linalg import norm
from numpy import array,mat,sin,cos,dot,eye
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from random import randint
import random
from icp.icp import icp
from read_data import *



Folder = '/home/dinesh/CarCrash/data/Fifth/'


def distance(point_1,point_2):
    return np.sqrt(np.mean((point_1-point_2)**2))


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
            if final_error_sample < 100:
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


def triangulate(P_1,P_2,keypoint_1,keypoint_2):
    point_3d_kp_hom = cv2.triangulatePoints(P_1,P_2,keypoint_1,keypoint_2)
    point_3d_kp_hom = point_3d_kp_hom.astype(np.float)
    point_3d_kp = point_3d_kp_hom / np.tile(point_3d_kp_hom[-1, :], (4, 1))
    return point_3d_kp

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

def check_nviews_new(RT_all,K_all,all_bb,cam_index,kp_all,point_3d_kp_all):
    count_views = 0
    camera = []
    cameras = []
    for i,boundingbox in enumerate(all_bb):
        count = 0
        num_of_point = 0
        cam_num = 0

        for j,location_3d in enumerate(point_3d_kp_all):

            if len(location_3d) > 1:
                cam_num += 1
                P = np.dot(K_all[i],  RT_all[i])
                point_kp = kp_all[i][j]
                point_2d = np.dot(P,location_3d)
                point_2d = point_2d/np.tile(point_2d[-1, :], (3, 1))
                if point_2d[0] > boundingbox[0] - 20 and point_2d[0] < boundingbox[0] + boundingbox[2] + 20 and point_2d[1] > boundingbox[1] - 20 and point_2d[1] < boundingbox[1] + boundingbox[3] + 20:
                    num_of_point = num_of_point + 1

        
        if num_of_point > cam_num*7/10:
            cameras.append(i)
            camera.append(cam_index[i])
    for l,cams_l in enumerate(cameras):
        for h,cams_h in enumerate(cameras):
            P_1 = np.dot(K_all[cams_l],  RT_all[cams_l]) 
            P_2 = np.dot(K_all[cams_h],  RT_all[cams_h])

            if l != h:
                kp_1 =  kp_all[cams_l]
                kp_2 = kp_all[cams_h]
                for kp_i,kp_v in enumerate(kp_1):
                    if kp_1[kp_i,2] >10 and kp_2[kp_i,2] >10:
                        keypoint_1 = kp_1[kp_i,0:2].reshape(2,1).astype(np.float)
                        keypoint_2 = kp_2[kp_i,0:2].reshape(2,1).astype(np.float)
                        point_3d_kp = triangulate(P_1,P_2,keypoint_1,keypoint_2)
                        point_3d_kp_all[kp_i] = (point_3d_kp_all[kp_i]+point_3d_kp)/2
    
    count_views = 0
    camera = []
    cameras = []
    for i,boundingbox in enumerate(all_bb):
        count = 0
        num_of_point = 0
        cam_num = 0

        for j,location_3d in enumerate(point_3d_kp_all):

            if len(location_3d) > 1:
                cam_num += 1
                P = np.dot(K_all[i],  RT_all[i])
                point_kp = kp_all[i][j]
                point_2d = np.dot(P,location_3d)
                point_2d = point_2d/np.tile(point_2d[-1, :], (3, 1))
                #print(Reproject_error_1view(P,point_kp,location_3d))
                if point_2d[0] > boundingbox[0] - 20 and point_2d[0] < boundingbox[0] + boundingbox[2] + 20 and point_2d[1] > boundingbox[1] - 20 and point_2d[1] < boundingbox[1] + boundingbox[3] + 20:
                    num_of_point = num_of_point + 1
                #print(distance(point_2d,point_kp))
                #if Reproject_error_1view(P,point_kp,location_3d)<100:
                #    print(Reproject_error_1view(P,point_kp,location_3d),'s')
        
        if num_of_point > cam_num*7/10:
            cameras.append(i)
            camera.append(cam_index[i])
    
    return len(np.unique(camera)),cameras,point_3d_kp_all


def ransac_nview(all_bb,kp_all,cam_index,RT_all,K_all):
    ransac_error = 100000
    ransac_num = 0
    location_3d = [0,0,0,0]
    bb_inside_final = []
    for loop in range(2000):
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
                #reproject_error_kp += Reproject_error_2view(P_1,P_2,keypoint_1,keypoint_2,point_3d_kp)
                reproject_error_kp += Reproject_error_nview(RT_all,K_all,all_bb,kp_all,kp_i,cam_index,point_3d_kp)
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
        
        #reproject_error = Reproject_error_2view(P_1,P_2,center_bb,center_bb_compare,point_3d)
        reproject_error = reproject_error_kp# + reproject_error_kp
        #reproject_error = Reproject_error(RT_all,K_all,all_bb,cam_index,point_3d)
        num_inside,bb_inside,point_3d_kp_all_loop = check_nviews_new(RT_all,K_all,all_bb,cam_index,kp_all,point_3d_kp_all_loop)
        if count > 2 and num_inside > ransac_num:
            ransac_num = num_inside
            print(reproject_error,count,bb_inside)
            #print(bb_inside)
            ransac_error = reproject_error
            #if :
            point_3d = point_3d_kp_sum/count
                #print(point_3d_kp_all_final)
            point_3d_kp_all_final = point_3d_kp_all_loop
            location_3d = point_3d
            ransac_num = num_inside
            bb_inside_final =bb_inside
                
      #    reproject_error
#        
#            print(reproject_error)
#            point_3d = point_3d_kp_sum/count
#            location_3d = point_3d
#            
    #print(location_3d)
    return point_3d_kp_all_final,location_3d,bb_inside_final

def find_matches(all_bb,kp_all,cam_index,RT_all,K_all,bb_counter,bb_final,point_3d_kp):
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
        #if point_2d[0] > boundingbox[0] and point_2d[0] < boundingbox[0] + boundingbox[2] and point_2d[1] > boundingbox[1] and point_2d[1] < boundingbox[1] + boundingbox[3]:
        if i in bb_final:
            # print()
            # print(np.sqrt(np.mean((point_2d[0:-1] - center_boundingbox)**2)))
            img = cv2.imread(Folder + str(cam_index[i]) + '/' + str(synched_images[cam_index[i]]).zfill(5) + '.png' )
            cv2.rectangle(img,(int(boundingbox[0]),int(boundingbox[1])),(int(boundingbox[0] + boundingbox[2]),int(boundingbox[1]+boundingbox[3])),c,3)
            cv2.circle(img,tuple(point_2d[0:2]),3,(255,0,128),5)
            for i,point_3d in enumerate(point_3d_kp):
                if len(point_3d) > 0:
                    #print(point_3d)
                    point = np.dot(P_1,point_3d)
                    point = point/np.tile(point[-1, :], (3, 1))
                    #print(point)
                    #cv2.circle(img,tuple(point[0:2]),3,(255,0,128),5)
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

## Read extrinsicsd
RT, RT_index = extrinsics(Folder)
## Read intrinsics\
K, K_index = intrinsics(Folder)
## Read cameraSync
diff = cameraSync(Folder)

time = 650

synched_images = - diff + time
## read keypoints

data = Read_keypoints(synched_images,Folder)

## find all boundix boxes
all_bb,kp_all,cam_index,RT_all,K_all = time_instance_data(data,synched_images,RT,RT_index,K,K_index)

### Ransac two view triangulation
bb_counter = 0
Car_3d = []
while 1:
    try:
        point_3d_kp,location_3d,bb_final = ransac_nview(all_bb,kp_all,cam_index,RT_all,K_all)
    except:
        break
    #print(point_3d_kp)
    print(bb_final)
    if len(bb_final) >2:
        Car_3d.append(point_3d_kp)
    else:
        break
    # find the cars across views
    all_bb,kp_all,cam_index,RT_all,K_all,bb_counter = find_matches(all_bb,kp_all,cam_index,RT_all,K_all,bb_counter,bb_final,point_3d_kp)
    
    print(bb_counter)
    #aSDASD

file = open(Folder + '/Car3D/' + str(time).zfill(5) +  '.txt','w') 
keypoints = np.loadtxt('/home/dinesh/CarCrash/codes/CollisionRecon/keypoints/car_cad/car1_kp.txt', dtype='f', delimiter=',')
for loop,car_points_3d in enumerate(Car_3d):
    
    
    # get the overlapping key points
    points_mesh = []
    points_ours = []
    for ind,points in enumerate(car_points_3d):
        if points == [] or ind == 8 or ind ==9:
            ind = ind
        else:
            points_ours.append(points)
            points_mesh.append(keypoints[ind])

    # find the scale between the mesh and the triangulated points
    scale_sum = 0
    coun = 0
    scale =[0] * 100
    for indices,points_3d in enumerate(points_ours):
        for indices_2,points_3d_2 in enumerate(points_ours):
            if indices_2 > indices and np.abs(np.mean(points_3d - points_3d_2))/np.abs(np.mean(points_mesh[indices] - points_mesh[indices_2])) < 5:
                scale[coun] = np.abs(np.mean(points_3d - points_3d_2))/np.abs(np.mean(points_mesh[indices] - points_mesh[indices_2]))
                coun += 1
    
    # ransac compute transformation
    scale_all = scale[0:coun]
    ransac_max = 10000
    for i,scale_check in enumerate(scale_all):
        #scale_check = np.mean(scale_all)
        points_ours_icp = np.array([[0,0,0]] * len(points_ours)).astype(np.float)
        points_mesh_icp = np.array([[0,0,0]] * len(points_ours)).astype(np.float)
        for indices,points_3d in enumerate(points_ours):
            points_ours_icp[indices] = points_ours[indices][0:3].T[0]
            points_mesh_icp[indices] = points_mesh[indices][0:3].T[0]*scale_check
        try:
            RT_transform,distance = icp(points_mesh_icp,points_ours_icp, None, 20, 0.001)
            if ransac_max > np.mean(distance) :
                #print(ransac_max,'s')
                #print(scale_check)
                ransac_max = np.mean(distance)
                RT_transform_best = RT_transform
                scale_best = scale_check
                

        except:
            print('icp failed')#scale = print(points_3d)
    #if 
    #print(RT_transform_best)
    print(len(points_ours))
    if ransac_max < 0.1:
        RT_transform_best = RT_transform_best
        file.write(str(scale_best))
        file.write(' ')
        file.write('/home/dinesh/CarCrash/codes/CollisionRecon/keypoints/car_cad/car1.obj')
        file.write(' ')
        file.write(str(RT_transform_best[0][0]))
        file.write(' ')
        file.write(str(RT_transform_best[0][1]))
        file.write(' ')
        file.write(str(RT_transform_best[0][2]))
        file.write(' ')
        file.write(str(RT_transform_best[0][3]))
        file.write(' ')
        file.write(str(RT_transform_best[1][0]))
        file.write(' ')
        file.write(str(RT_transform_best[1][1]))
        file.write(' ')
        file.write(str(RT_transform_best[1][2]))
        file.write(' ')
        file.write(str(RT_transform_best[1][3]))
        file.write(' ')
        file.write(str(RT_transform_best[2][0]))
        file.write(' ')
        file.write(str(RT_transform_best[2][1]))
        file.write(' ')
        file.write(str(RT_transform_best[2][2]))
        file.write(' ')
        file.write(str(RT_transform_best[2][3]))
        file.write(' ')
        file.write(str(RT_transform_best[3][0]))
        file.write(' ')
        file.write(str(RT_transform_best[3][1]))
        file.write(' ')
        file.write(str(RT_transform_best[3][2]))
        file.write(' ')
        file.write(str(RT_transform_best[3][3]))
        icp_error = np.dot(RT_transform_best,np.transpose(np.append(points_mesh_icp[0],1)))[0:3] - points_ours_icp[0]
        #print(points_ours_icp[0])
        #print(points_mesh_icp[0])
        #print(np.dot(RT_transform_best,np.transpose(np.append(points_mesh_icp[0],1))))
        #print(np.sqrt(np.mean(icp_error**2)))
        print(ransac_max)
        
        file.write('\n')
file.close()