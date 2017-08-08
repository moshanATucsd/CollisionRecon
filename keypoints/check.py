#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 20:55:33 2017

@author: dinesh
"""

 # get the overlapping key points
points_mesh = []
points_ours = []
for ind,points in enumerate(car_points_3d):
    if len(points) < 4 or ind == 8 or ind == 9 :
        ind = ind
    else:
        points_ours.append(points[0:3].reshape(1,3))
        points_mesh.append(keypoints[ind].astype(np.float))


#for p_index,pt in enumerate(points_ours):
#    print(pt)

# find the scale between the mesh and the triangulated points
scale_sum = 0
coun = 0
scale =[0] * 100
for indices,points_3d in enumerate(points_ours):
    #print(points_3d)
    for indices_2,points_3d_2 in enumerate(points_ours):
        if indices_2 > indices and np.abs(np.mean(points_3d - points_3d_2))/np.abs(np.mean(points_mesh[indices] - points_mesh[indices_2])) < 5:
            scale[coun] = np.abs(np.mean(points_3d - points_3d_2))/np.abs(np.mean(points_mesh[indices] - points_mesh[indices_2]))
            coun += 1
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
            #print(distance)
            icp_error = 0
            for k,asa in enumerate(points_mesh_icp):
                icp_error += np.mean((np.dot(RT_transform,np.transpose(np.append(points_mesh_icp[k],1)))[0:3] - points_ours_icp[k])**2)
            icp_error =  np.mean(distance)
            if ransac_max > np.mean(distance) :
                    #print(points_mesh_icp)
                    #print(points_ours_icp)
                print(icp_error)
                ransac_max = icp_error
                RT_transform_best = RT_transform
                scale_best = scale_check
                    
    
        except:
                print('icp failed')#scale = print(points_3d)
for k in range(10):
    print(np.dot(RT_transform_best,np.transpose(np.append(points_mesh_icp[k],1)))[0:3])
    print(points_ours_icp[k])
