#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 00:02:17 2017

@author: dinesh
"""

import numpy as np
from read_write_data import *
from check_cameras import *
from python_ceres.ba import *

npzfile = np.load('car_video.npz')
#npzfile = np.load('1.npy.npz')
a =npzfile['arr_0']
b =npzfile['arr_1']
c =npzfile['arr_2']
d =npzfile['arr_3']
e =npzfile['arr_4']
f =npzfile['arr_5']
g =npzfile['arr_6']
h =npzfile['arr_7']
#k = []
#k.append(a[0])
#p = []
#p.append(b[0])
#c = c[0:3]
#d = d[0:2]
#t =[]
#t.append(e[0])
#point_3d = c * f
#point_3d_final = np.dot(g,np.append(point_3d,1))
#print(point_3d_final)
#p_final = np.dot(k[0],p[0])
#p_2d = np.dot(p_final,np.append(point_3d_final,1))
#print(p_2d/p_2d[2])
#scale,RT_transform = car_fit_nview_with_ceres(k,p,c,d,e,f,g)

#print(e)
#scale,RT_transform = car_fit_nview_with_ceres(a,b,c,d,e,f,g[0:3])
#print(RT_transform)
print(g)
#scale,RT_transform = car_fit_video_nview_with_ceres(a,b,c,d,e,f,[g],e*0)
scale,RT_transform = car_fit_video_nview_with_ceres(a,b,c,d,e,f,g,h)
print(RT_transform)
# 950 924
# [-2.01338178  0.08241741  0.97774704  1.        ]
# [-0.09778239  0.16383822 -0.05076314  1.        ]
