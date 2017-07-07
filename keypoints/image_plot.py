#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 17:25:30 2017

@author: dinesh
"""
import cv2

def drawCar(img,keypoints,bb=None):
#    if  keypoints[0,2] >50 and keypoints[2,2] >50:
#        cv2.line(img,tuple(keypoints[0,0:2]),tuple(keypoints[2,0:2]),(255,0,0),5)
#    if  keypoints[1,2] >50 and keypoints[3,2] >50:
#        cv2.line(img,tuple(keypoints[1,0:2]),tuple(keypoints[3,0:2]),(0,0,255),5)
#    if  keypoints[0,2] >50 and keypoints[1,2] >50:
#        cv2.line(img,tuple(keypoints[0,0:2]),tuple(keypoints[1,0:2]),(0,255,0),5)
#    if  keypoints[3,2] >50 and keypoints[2,2] >50:
#        cv2.line(img,tuple(keypoints[3,0:2]),tuple(keypoints[2,0:2]),(128,128,0),5)
    threshold = 20
    # wheels
    if  keypoints[1,2] >threshold:
        cv2.circle(img,tuple(keypoints[1,0:2]),3,(0,255,0),3)
    if  keypoints[2,2] >threshold:
        cv2.circle(img,tuple(keypoints[2,0:2]),3,(64,255,255),3)
    if  keypoints[3,2] >threshold:
        	cv2.circle(img,tuple(keypoints[3,0:2]),3,(128,255,128),3)
    if  keypoints[0,2] >threshold:
        	cv2.circle(img,tuple(keypoints[0,0:2]),3,(128,255,0),3)

    if  keypoints[0,2] >threshold and keypoints[2,2] >threshold:
        cv2.line(img,tuple(keypoints[0,0:2]),tuple(keypoints[2,0:2]),(255,0,0),2)
    if  keypoints[1,2] >threshold and keypoints[3,2] >threshold:
        cv2.line(img,tuple(keypoints[1,0:2]),tuple(keypoints[3,0:2]),(255,0,0),2)
    if  keypoints[0,2] >threshold and keypoints[1,2] >threshold:
        cv2.line(img,tuple(keypoints[0,0:2]),tuple(keypoints[1,0:2]),(255,0,0),2)
    if  keypoints[2,2] >threshold and keypoints[3,2] >threshold:
        cv2.line(img,tuple(keypoints[2,0:2]),tuple(keypoints[3,0:2]),(255,0,0),2)



    # top of car
    if  keypoints[10,2] >threshold:
        	cv2.circle(img,tuple(keypoints[10,0:2]),3,(255,128,128),3)
    if  keypoints[11,2] >threshold:
        	cv2.circle(img,tuple(keypoints[11,0:2]),3,(128,128,128),3)
    if  keypoints[12,2] >threshold:
        	cv2.circle(img,tuple(keypoints[12,0:2]),3,(0,128,255),3)
    if  keypoints[13,2] >threshold:
        	cv2.circle(img,tuple(keypoints[13,0:2]),3,(0,255,255),3)

    if  keypoints[10,2] >threshold and keypoints[12,2] >threshold:
        cv2.line(img,tuple(keypoints[10,0:2]),tuple(keypoints[12,0:2]),(0,255,0),2)
    if  keypoints[11,2] >threshold and keypoints[13,2] >threshold:
        cv2.line(img,tuple(keypoints[11,0:2]),tuple(keypoints[13,0:2]),(0,255,0),2)
    if  keypoints[10,2] >threshold and keypoints[11,2] >threshold:
        cv2.line(img,tuple(keypoints[10,0:2]),tuple(keypoints[11,0:2]),(0,255,0),2)
    if  keypoints[12,2] >threshold and keypoints[13,2] >threshold:
        cv2.line(img,tuple(keypoints[12,0:2]),tuple(keypoints[13,0:2]),(0,255,0),2)
        
    # front head lights
    if  keypoints[4,2] >threshold:
        	cv2.circle(img,tuple(keypoints[4,0:2]),3,(0,255,0),3)
    if  keypoints[0,2] >threshold and keypoints[4,2] >threshold:
        cv2.line(img,tuple(keypoints[0,0:2]),tuple(keypoints[4,0:2]),(0,0,255),2)
    if  keypoints[10,2] >threshold and keypoints[4,2] >threshold:
        cv2.line(img,tuple(keypoints[10,0:2]),tuple(keypoints[4,0:2]),(0,0,255),2)

    if  keypoints[5,2] >threshold:
        cv2.circle(img,tuple(keypoints[5,0:2]),3,(128,0,0),3)
    if  keypoints[1,2] >threshold and keypoints[5,2] >threshold:
        cv2.line(img,tuple(keypoints[1,0:2]),tuple(keypoints[5,0:2]),(0,0,255),2)
    if  keypoints[11,2] >threshold and keypoints[5,2] >threshold:
        cv2.line(img,tuple(keypoints[11,0:2]),tuple(keypoints[5,0:2]),(0,0,255),2)
    if  keypoints[4,2] >threshold and keypoints[5,2] >threshold:
        cv2.line(img,tuple(keypoints[4,0:2]),tuple(keypoints[5,0:2]),(0,0,255),2)

    # back head lights
    if  keypoints[6,2] >threshold:
        	cv2.circle(img,tuple(keypoints[6,0:2]),3,(255,0,0),3)
    if  keypoints[2,2] >threshold and keypoints[6,2] >threshold:
        cv2.line(img,tuple(keypoints[2,0:2]),tuple(keypoints[6,0:2]),(255,0,255),2)
    if  keypoints[12,2] >threshold and keypoints[6,2] >threshold:
        cv2.line(img,tuple(keypoints[12,0:2]),tuple(keypoints[6,0:2]),(255,0,255),2)

    
    if  keypoints[7,2] >threshold:
        	cv2.circle(img,tuple(keypoints[7,0:2]),3,(255,0,128),5)
    if  keypoints[3,2] >threshold and keypoints[7,2] >threshold:
        cv2.line(img,tuple(keypoints[3,0:2]),tuple(keypoints[7,0:2]),(255,0,255),2)
    if  keypoints[13,2] >threshold and keypoints[7,2] >threshold:
        cv2.line(img,tuple(keypoints[13,0:2]),tuple(keypoints[7,0:2]),(255,0,255),2)
    if  keypoints[6,2] >threshold and keypoints[7,2] >threshold:
        cv2.line(img,tuple(keypoints[6,0:2]),tuple(keypoints[7,0:2]),(255,0,255),2)

    # mirrror
    if  keypoints[8,2] >threshold:
        	cv2.circle(img,tuple(keypoints[8,0:2]),3,(128,0,128),5)
    if  keypoints[8,2] >threshold and keypoints[4,2] >threshold:
        cv2.line(img,tuple(keypoints[8,0:2]),tuple(keypoints[4,0:2]),(0,0,255),2)
    
    if  keypoints[9,2] >threshold:
        cv2.circle(img,tuple(keypoints[9,0:2]),3,(0,128,128),5)
    if  keypoints[9,2] >threshold and keypoints[5,2] >threshold:
        cv2.line(img,tuple(keypoints[9,0:2]),tuple(keypoints[5,0:2]),(0,0,255),2)
    
    #cv2.line(img,tuple(keypoints[0,0:2]),tuple(keypoints[2,0:2]),(255,0,0),1)
    #cv2.line(img,tuple(keypoints[4,0:2]),tuple(keypoints[6,0:2]),(0,255,0),1)
    #cv2.line(img,tuple(keypoints[5,0:2]),tuple(keypoints[7,0:2]),(0,255,0),1)
    #cv2.line(img,tuple(keypoints[4,0:2]),tuple(keypoints[5,0:2]),(0,255,0),1)
    #cv2.line(img,tuple(keypoints[6,0:2]),tuple(keypoints[7,0:2]),(0,255,0),1)

    #cv2.line(img,tuple(keypoints[1]),tuple(keypoints[2]),(0,255,0),5)
    #cv2.line(img,tuple(keypoints[2]),tuple(keypoints[3]),(0,255,0),5)
    #cv2.line(img,tuple(keypoints[0]),tuple(keypoints[3]),(0,255,0),5)
    #print(keypoints[0]- [20,20])


#srinivasa
#    box_width = int(max(bb[i][2],bb[i][3])/10)
#    if  keypoints[0,2] >50:
#        cv2.rectangle(img,tuple(keypoints[0,0:2].astype(np.int) - [box_width,box_width]) ,tuple(keypoints[0,0:2].astype(np.int) + [box_width,box_width]),(128,0,128), thickness=2)
#        
#    if  keypoints[1,2] >50:
#        cv2.rectangle(img,tuple(keypoints[1,0:2].astype(np.int) - [box_width,box_width]) ,tuple(keypoints[1,0:2].astype(np.int) + [box_width,box_width]),(255,0,255), thickness=2)
#    if  keypoints[2,2] >50:
#        cv2.rectangle(img,tuple(keypoints[2,0:2].astype(np.int) - [box_width,box_width]) ,tuple(keypoints[2,0:2].astype(np.int) + [box_width,box_width]),(0,255,128), thickness=2)
#    if  keypoints[3,2] >50:
#        cv2.rectangle(img,tuple(keypoints[3,0:2].astype(np.int) - [box_width,box_width]) ,tuple(keypoints[3,0:2].astype(np.int) + [box_width,box_width]),(0,128,128), thickness=2)

#    cv2.line(img,tuple(keypoints[4]),tuple(keypoints[5]),(255,0,0),5)
#    cv2.line(img,tuple(keypoints[5]),tuple(keypoints[6]),(255,0,0),5)
#    cv2.line(img,tuple(keypoints[6]),tuple(keypoints[7]),(255,0,0),5)
#   cv2.line(img,tuple(keypoints[7]),tuple(keypoints[4]),(255,0,0),5)


#    cv2.line(img,tuple(keypoints[10]),tuple(keypoints[11]),(255,0,0),5)
#    cv2.line(img,tuple(keypoints[11]),tuple(keypoints[12]),(255,0,0),5)
#    cv2.line(img,tuple(keypoints[12]),tuple(keypoints[13]),(255,0,0),5)
#    cv2.line(img,tuple(keypoints[13]),tuple(keypoints[10]),(255,0,0),5)
    
#    cv2.line(img,tuple(keypoints[10]),tuple(keypoints[11]),(255,0,0),5)
#    cv2.line(img,tuple(keypoints[11]),tuple(keypoints[12]),(255,0,0),5)
#    cv2.line(img,tuple(keypoints[12]),tuple(keypoints[13]),(255,0,0),5)
#    cv2.line(img,tuple(keypoints[13]),tuple(keypoints[10]),(255,0,0),5)

    
#    cv2.line(img,tuple(keypoints[3]),tuple(keypoints[4]),(255,0,0),5)
#    cv2.line(img,tuple(keypoints[9]),tuple(keypoints[10]),(255,0,0),5)
#    cv2.line(img,tuple(keypoints[10]),tuple(keypoints[11]),(255,0,0),5)
#    cv2.line(img,tuple(keypoints[11]),tuple(keypoints[12]),(255,0,0),5)
#    cv2.line(img,tuple(keypoints[12]),tuple(keypoints[13]),(255,0,0),5)


#minh
#    cv2.line(img,tuple(keypoints[3]),tuple(keypoints[3]),(255,0,0),5)
#    cv2.line(img,tuple(keypoints[1]),tuple(keypoints[4]),(255,0,0),5)
#    cv2.line(img,tuple(keypoints[4]),tuple(keypoints[5]),(255,0,0),5)
#    cv2.line(img,tuple(keypoints[9]),tuple(keypoints[6]),(255,0,0),5)
#    cv2.line(img,tuple(keypoints[8]),tuple(keypoints[7]),(255,0,0),5)
#    cv2.line(img,tuple(keypoints[10]),tuple(keypoints[8]),(255,0,0),5)

def drawPerson(img,keypoints,bb=None):
    threshold = 10
    # wheels
    if  keypoints[0,2] >threshold:
        	cv2.circle(img,tuple(keypoints[0,0:2]),3,(128,255,0),2)
    if  keypoints[1,2] >threshold:
        cv2.circle(img,tuple(keypoints[1,0:2]),3,(0,255,0),2)
    if  keypoints[2,2] >threshold:
        cv2.circle(img,tuple(keypoints[2,0:2]),3,(64,255,255),2)
    if  keypoints[3,2] >threshold:
        	cv2.circle(img,tuple(keypoints[3,0:2]),3,(128,255,128),2)
    if  keypoints[4,2] >threshold:
        	cv2.circle(img,tuple(keypoints[4,0:2]),3,(0,255,0),2)
    if  keypoints[5,2] >threshold:
        cv2.circle(img,tuple(keypoints[5,0:2]),3,(128,0,0),2)
    if  keypoints[6,2] >threshold:
        	cv2.circle(img,tuple(keypoints[6,0:2]),3,(255,0,0),2)
    if  keypoints[7,2] >threshold:
        	cv2.circle(img,tuple(keypoints[7,0:2]),3,(255,0,128),2)
    if  keypoints[8,2] >threshold:
        	cv2.circle(img,tuple(keypoints[8,0:2]),3,(128,0,128),2)
    if  keypoints[9,2] >threshold:
        cv2.circle(img,tuple(keypoints[9,0:2]),3,(0,128,128),2)
    if  keypoints[10,2] >threshold:
        	cv2.circle(img,tuple(keypoints[10,0:2]),3,(255,128,128),2)
    if  keypoints[11,2] >threshold:
        	cv2.circle(img,tuple(keypoints[11,0:2]),3,(128,128,128),2)
    if  keypoints[12,2] >threshold:
        	cv2.circle(img,tuple(keypoints[12,0:2]),3,(0,128,255),2)
    if  keypoints[13,2] >threshold:
        	cv2.circle(img,tuple(keypoints[13,0:2]),3,(0,255,255),2)
    if  keypoints[14,2] >threshold:
        	cv2.circle(img,tuple(keypoints[14,0:2]),3,(0,255,64),2)
    if  keypoints[15,2] >threshold:
        	cv2.circle(img,tuple(keypoints[15,0:2]),3,(0,64,255),2)

    if  keypoints[0,2] >threshold and keypoints[1,2] >threshold:
        cv2.line(img,tuple(keypoints[0,0:2]),tuple(keypoints[1,0:2]),(0,255,0),2)
    if  keypoints[1,2] >threshold and keypoints[2,2] >threshold:
        cv2.line(img,tuple(keypoints[1,0:2]),tuple(keypoints[2,0:2]),(0,255,0),2)
    if  keypoints[2,2] >threshold and keypoints[6,2] >threshold:
        cv2.line(img,tuple(keypoints[2,0:2]),tuple(keypoints[6,0:2]),(0,255,0),2)
    if  keypoints[3,2] >threshold and keypoints[4,2] >threshold:
        cv2.line(img,tuple(keypoints[3,0:2]),tuple(keypoints[4,0:2]),(0,255,0),2)
    if  keypoints[3,2] >threshold and keypoints[6,2] >threshold:
        cv2.line(img,tuple(keypoints[3,0:2]),tuple(keypoints[6,0:2]),(0,255,0),2)
    if  keypoints[4,2] >threshold and keypoints[5,2] >threshold:
        cv2.line(img,tuple(keypoints[4,0:2]),tuple(keypoints[5,0:2]),(0,255,0),2)
    if  keypoints[6,2] >threshold and keypoints[8,2] >threshold:
        cv2.line(img,tuple(keypoints[6,0:2]),tuple(keypoints[8,0:2]),(0,255,0),2)
    if  keypoints[8,2] >threshold and keypoints[9,2] >threshold:
        cv2.line(img,tuple(keypoints[8,0:2]),tuple(keypoints[9,0:2]),(0,255,0),2)
    if  keypoints[13,2] >threshold and keypoints[8,2] >threshold:
        cv2.line(img,tuple(keypoints[13,0:2]),tuple(keypoints[8,0:2]),(0,255,0),2)
    if  keypoints[10,2] >threshold and keypoints[11,2] >threshold:
        cv2.line(img,tuple(keypoints[10,0:2]),tuple(keypoints[11,0:2]),(0,255,0),2)
    if  keypoints[11,2] >threshold and keypoints[12,2] >threshold:
        cv2.line(img,tuple(keypoints[11,0:2]),tuple(keypoints[12,0:2]),(0,255,0),2)
    if  keypoints[12,2] >threshold and keypoints[8,2] >threshold:
        cv2.line(img,tuple(keypoints[12,0:2]),tuple(keypoints[8,0:2]),(0,255,0),2)
    if  keypoints[13,2] >threshold and keypoints[14,2] >threshold:
        cv2.line(img,tuple(keypoints[13,0:2]),tuple(keypoints[14,0:2]),(0,255,0),2)
    if  keypoints[14,2] >threshold and keypoints[15,2] >threshold:
        cv2.line(img,tuple(keypoints[14,0:2]),tuple(keypoints[15,0:2]),(0,255,0),2)
        