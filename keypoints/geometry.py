#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 17:07:39 2017

@author: dinesh
"""
import numpy as np
import cv2
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

def triangulate_visible_keypoints(P_1,P_2,kp_1,kp_2):
        #reproject_error_kp = 0
        point_3d_kp_all =[]
        for kp_i,kp_v in enumerate(kp_1):
            if kp_1[kp_i,2] >50 and kp_2[kp_i,2] >50:
                keypoint_1 = kp_1[kp_i,0:2].reshape(2,1).astype(np.float)
                keypoint_2 = kp_2[kp_i,0:2].reshape(2,1).astype(np.float)
                point_3d_kp = triangulate(P_1,P_2,keypoint_1,keypoint_2)
                point_3d_kp_all.append(point_3d_kp)
                # reproject_error_kp += Reproject_error_2view(P_1,P_2,keypoint_1,keypoint_2,point_3d_kp)
                # reproject_error_kp += Reproject_error_nview(RT_all,K_all,all_bb,kp_all,kp_i,cam_index,point_3d_kp)
            else:
                point_3d_kp_all.append([])
        return point_3d_kp_all



def check_withbb(location_3d,P,boundingbox):
    point_2d = np.dot(P,location_3d)
    point_2d = point_2d/np.tile(point_2d[-1, :], (3, 1))
    if point_2d[0] > boundingbox[0] - 20 and point_2d[0] < boundingbox[0] + boundingbox[2] + 20 and point_2d[1] > boundingbox[1] - 20 and point_2d[1] < boundingbox[1] + boundingbox[3] + 20:
        return 1
    return 0

def scale_transformation(car_points_3d,keypoints):
        #car_points_3d_center = [[0],[0],[0],[1]]
        #for 
       # print(np.mean(car_points_3d))
        
        # get the overlapping key points
        points_mesh = []
        points_ours = []
        for ind,points in enumerate(car_points_3d):
            if len(points) < 4 or ind == 8 or ind == 9 :
                #print(keypoints[ind])
                #print(points)
                ind = ind
            else:
                points_ours.append(points[0:3].reshape(1,3))
                points_mesh.append(keypoints[ind].astype(np.float))
        #print(points_ours)
        #print(points_mesh)
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
        scale_best = 1
        RT_transform_best = np.zeros([4,4])
        for i,scale_check in enumerate(scale_all):
            #scale_check = np.mean(scale_all)
            points_ours_icp = np.array([[0,0,0]] * len(points_ours)).astype(np.float)
            points_mesh_icp = np.array([[0,0,0]] * len(points_ours)).astype(np.float)
            for indices,points_3d in enumerate(points_ours):
                points_ours_icp[indices] = points_ours[indices]
                points_mesh_icp[indices] = points_mesh[indices]*scale_check
            try:
                RT_transform,distance = icp(points_mesh_icp,points_ours_icp, None, 200, 0.0001)
                icp_error = 0
                for k,asa in enumerate(points_mesh_icp):
                    icp_error += np.mean((np.dot(RT_transform,np.transpose(np.append(points_mesh_icp[k],1)))[0:3] - points_ours_icp[k])**2)
                if ransac_max > icp_error:#np.mean(distance) :
                    #print(points_mesh_icp)
                    #print(points_ours_icp)
                    #print(icp_error)
                    ransac_max = icp_error
                    RT_transform_best = RT_transform
                    scale_best = scale_check
                    
    
            except:
                print('icp failed')#scale = print(points_3d)
#        print(np.dot(RT_transform_best,np.transpose(np.append(points_mesh_icp[k],1)))[0:3])
#        print(points_ours_icp[k])
        return scale_best,RT_transform_best


def scale_transformation_without_icp(car_points_3d,keypoints):

        # get the overlapping key points
        points_mesh = []
        points_ours = []
        for ind,points in enumerate(car_points_3d):
            if len(points) < 4 or ind == 8 or ind == 9 :
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
        
        return scale_best,RT_transform_best





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




def count_inliers(all_bb,point_3d_kp_all,K_all,RT_all,kp_all,cam_index):
    camera = []
    cameras = []

    for i,boundingbox in enumerate(all_bb):
        num_of_point = 0
        cam_num = 0
        for j,location_3d in enumerate(point_3d_kp_all):
            if len(location_3d) > 1:
                cam_num += 1
                P = np.dot(K_all[i],  RT_all[i])
                #point_kp = kp_all[i][j]
                num_of_point += check_withbb(location_3d,P,boundingbox)
        
        if num_of_point > cam_num*8/10:
            cameras.append(i)
            camera.append(cam_index[i])
    return camera,cameras

def count_inliers_point(all_bb,location_3d,K_all,RT_all):
    num_of_point = 0
    if len(location_3d) > 1:
        for i,boundingbox in enumerate(all_bb):
            P = np.dot(K_all[i],  RT_all[i])
            num_of_point += check_withbb(location_3d,P,boundingbox)
            
    return num_of_point
def reproj_inliers_point(all_bb,location_3d,K_all,RT_all,kp_all,t,cameras):
    error = 0
    for i,boundingbox in enumerate(all_bb):
        if len(location_3d) > 1 and i in cameras:
            P = np.dot(K_all[i],  RT_all[i])
            point_2d_kp = kp_all[i][t]
            point_2d = np.dot(P,location_3d)    
            point_2d = point_2d/np.tile(point_2d[-1, :], (3, 1))
            error += distance(point_2d_kp,point_2d)
    if error == 0:
        error = 10000
    return error

def drawlines(img1,img2,lines,pt2):
    print(img1.shape)
    r,c,h = img1.shape
    r = lines
    print(lines[1][0])
    color = tuple(np.random.randint(0,255,3).tolist())
    x0,y0 = map(int, [0, -r[2]/r[1] ])
    x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
    img2 = cv2.line(img2, (x0,y0), (x1,y1), color,1)
    img1 = cv2.circle(img1,tuple([150,600]),5,color,5)
    return img1,img2

def triangulate(P_1,P_2,keypoint_1,keypoint_2):
    point_3d_kp_hom = cv2.triangulatePoints(P_1,P_2,keypoint_1,keypoint_2)
    point_3d_kp_hom = point_3d_kp_hom.astype(np.float)
    point_3d_kp = point_3d_kp_hom / np.tile(point_3d_kp_hom[-1, :], (4, 1))
    return point_3d_kp

        
def fundamental_matrix_check():
        K_1 = K_all[bb_index]
        K_2 = K_all[bb_compare_index]
        RT_1 = np.append(RT_all[bb_index],[0,0,0,1])
        RT_2 = np.append(RT_all[bb_compare_index],[0,0,0,1])
        RT_1 = np.reshape(RT_1,[4,4])
        RT_2 = np.asmatrix(np.reshape(RT_2,[4,4]))
        RT_1_inv = np.linalg.inv(RT_1)
        RT_between = np.dot(RT_1_inv,RT_2)
        print(RT_between)
        print(RT_between[0:3,3])
        #print(np.transpose(np.linalg.inv(K_2)))
        #print(np.linalg.inv(np.transpose(K_2)))
        KRK = np.dot(np.dot(np.linalg.inv(np.transpose(K_2)),RT_between[0:3,0:3]),np.transpose(K_1))
        KRT = np.dot(K_1,np.dot(np.transpose(RT_between[0:3,0:3]),RT_between[0:3,3]))
        KRT_3 = np.asmatrix([[0,-KRT[2],KRT[1]],[KRT[2],0,-KRT[0]],[-KRT[1],KRT[0],0]])
        print(KRT_3)
        print(KRT)
        
        F = np.dot(KRK,KRT_3)
        img_1 = cv2.imread(Folder + str(cam_index[bb_index]) + '/' + str(synched_images[cam_index[bb_index]]).zfill(5) + '.png' )
        img_2 = cv2.imread(Folder + str(cam_index[bb_compare_index]) + '/' + str(synched_images[cam_index[bb_compare_index]]).zfill(5) + '.png' )
        pt2 = [[150],[600],[1]]
        l = F*pt2
        img1,img2 = drawlines(img_1,img_2,l,pt2)
        cv2.imwrite(str(1000) + '.png',img1)
        cv2.imwrite(str(1001) + '.png',img2)

        print(F)
def prun_bb(all_bb,kp_all,cam_index,RT_all,K_all,bb_counter,bb_final,point_3d_kp):
    all_bb_new = []
    cam_index_new = []
    RT_all_new = []
    K_all_new =[]
    kp_all_new = []
    c = []
    for i,boundingbox in enumerate(all_bb):
        if i in bb_final:
            i = i
        else:
            all_bb_new.append(boundingbox)
            cam_index_new.append(cam_index[i])
            RT_all_new.append(RT_all[i])
            K_all_new.append(K_all[i])
            kp_all_new.append(kp_all[i])
    return all_bb_new,kp_all_new,cam_index_new,RT_all_new,K_all_new,bb_counter

