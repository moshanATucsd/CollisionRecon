#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 17:11:44 2017

@author: dinesh
"""

mutable_object ={}

def on_press(event):
    print('you pressed', event.button, event.xdata, event.ydata)
    plt.close()
    mutable_object['key'] =  [event.xdata, event.ydata]  
    return([event.xdata, event.ydata])

def keypoint_press(filename):
    keypoints = []
    for i in range(1):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = Image.open(filename)
        arr = np.asarray(im)
        plt_image=plt.imshow(arr)
        cid = fig.canvas.mpl_connect('button_press_event', on_press)
        #rs=widgets.RectangleSelector(
        #    ax, onselect, drawtype='box',
        #    rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=True))
        plt.show()
        keypoints.append(mutable_object['key'])
    print(keypoints)
    return(keypoints)

def check_camera_params(K_all,RT_all,all_bb,time):
    point_3d_kp_all = []
    keypoints_1 = keypoint_press(Folder + str(0) + '/' + str(synched_images[0]).zfill(5) + '.png')
    print(keypoints_1)
    keypoints_2 = keypoint_press(Folder + str(1) + '/' + str(synched_images[1]).zfill(5) + '.png')
    #print(point_2d_2)
    
    P_1 = np.dot(K_all[0],  RT_all[0]) 
    P_2 = np.dot(K_all[4],  RT_all[4])
    for li,kp_1 in enumerate(keypoints_1):
        keypoint_1 = np.array(kp_1)
        keypoint_2 = np.array(keypoints_2[li]);
        point_3d_kp_all.append(triangulate(P_1,P_2,keypoint_1,keypoint_2))
#    keypoint_1 = np.array([1654.8,760.2]);
#    keypoint_2 = np.array([1055.25,610.25]);
#    point_3d_kp_all.append(triangulate(P_1,P_2,keypoint_1,keypoint_2))
#    keypoint_1 = np.array([690.89,428.04]);
#    keypoint_2 = np.array([1825.35,308.7]);
#    point_3d_kp_all.append(triangulate(P_1,P_2,keypoint_1,keypoint_2))
#    keypoint_1 = np.array([547.9011,921.0741]);
#    keypoint_2 = np.array([743.7500,613.2500]);
#    point_3d_kp_all.append(triangulate(P_1,P_2,keypoint_1,keypoint_2))
    point_3d_kp_all.append([[1.1954653612647386],[ -0.9993841072343063],[ 6.7841787420247686],[1]])  

#    
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
