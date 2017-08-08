import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
from image_plot import *



Folder = '/home/dinesh/CarCrash/data/Fifth/'
Folder = '/home/dinesh/CarCrash/data/CarCrash/Cleaned/'
Folder = '/home/dinesh/CarCrash/data/syn/'
Folder = '/home/dinesh/CarCrash/data/Kitti_1/'
Folder = '/home/dinesh/CarCrash/data/test/'


    # front head lights
def label_bb(img,kp,bb,bb_loop):
    if class_name[bb_loop] == 'car':
        cv2.rectangle(img,(int(bb[bb_loop][0]),int(bb[bb_loop][1])),(int(bb[bb_loop][0] + bb[bb_loop][2]),int(bb[bb_loop][1]+bb[bb_loop][3])),(0,255,0))
        drawCar(img,kp,bb)
    if class_name[bb_loop] == 'person':
        cv2.rectangle(img,(int(bb[bb_loop][0]),int(bb[bb_loop][1])),(int(bb[bb_loop][0] + bb[bb_loop][2]),int(bb[bb_loop][1]+bb[bb_loop][3])),(255,0,0))
        drawPerson(img,kp,bb)
    if class_name[bb_loop] == 'truck':
        cv2.rectangle(img,(int(bb[bb_loop][0]),int(bb[bb_loop][1])),(int(bb[bb_loop][0] + bb[bb_loop][2]),int(bb[bb_loop][1]+bb[bb_loop][3])),(0,0,255))
        drawCar(img,kp,bb)
    if class_name[bb_loop] == 'bus':
        cv2.rectangle(img,(int(bb[bb_loop][0]),int(bb[bb_loop][1])),(int(bb[bb_loop][0] + bb[bb_loop][2]),int(bb[bb_loop][1]+bb[bb_loop][3])),(255,0,255))
        drawCar(img,kp,bb)

for main_loop in range(1,21):
    filenames = sorted(glob.glob(Folder + str(main_loop-1) + '/keypoints_txt/*.txt'))
    for index,name in enumerate(filenames):
        bb = []
        points = []
        class_name = []
        img_name = name.split('keypoints_txt')[0] + name.split('keypoints_txt')[1].split('.txt')[0]
        img_original = cv2.imread(img_name)
        img_instance_segment = cv2.imread(img_name.replace('//','/labelled/'))
        img = img_instance_segment*0#img_original*0
        img_final = img_original
        print(img_name)
        with open(name) as f:
            lines = f.readlines()
        for line in lines:
            bb.append(np.array(line.split(',')[1:5]).astype(np.float))
            points.append(np.array(line.split(',')[5:-1]).astype(np.float))
            class_name.append(line.split(',')[-1].split('\n')[0]) 
        for bb_loop,bb_num in enumerate(bb):
            points_array = np.array(points[bb_loop])#.splitlines()[0].split(','))
            points_arranged = points_array.reshape(int(len(points_array)/3),3)
            kp = points_arranged[:,0:3]
            kp = np.round(kp.astype(np.float)).astype(np.int)
            kp[:,0] = bb[bb_loop][0] + kp[:,0]*(bb[bb_loop][2]/64)
            kp[:,1] = bb[bb_loop][1] + kp[:,1]*(bb[bb_loop][3]/64)
            kp[:,2] = np.round(points_arranged[:,2].astype(np.float)*100).astype(np.int)
            label_bb(img,kp,bb,bb_loop)
            label_bb(img_original,kp,bb,bb_loop)
        #img_original
        cv2.addWeighted(img_original, 1, img_instance_segment, 0.5, 0, img_final)
        cv2.addWeighted(img, 1, img_instance_segment, 0.5, 0, img)
        #cv2.addWeighted(img_original, 1, img, 10, 0, img_final)
        
        #cv2.addWeighted(img_final, 1, img_final, 1, 0, img_final)
        cv2.imwrite(img_name.replace('//','/labelled_image/'), img_final)
        cv2.imwrite(img_name.replace('//','/labelled_image/b_'), img)
#with open('/home/dinesh/CarCrash/data/Fifth/bb_all.txt') as f:
##with open('/home/dinesh/CarCrash/data/Fifth/bb_person.txt') as f:
    #lines = f.readlines()
#names = []
#bb = []
#for line in lines:
    ##print(line)
    #names.append(line.split(' ')[0])
    #bb.append(np.array(line.split(' ')[1:5]).astype(np.float))

#with open('/home/dinesh/CarCrash/data/Fifth/keypoints_all.txt') as f:
##with open('/home/dinesh/CarCrash/data/Fifth/keypoints_person.txt') as f:
    #keypoints = f.readlines()

#points = []
#for point in keypoints:
    ##print(line)
    #points.append(point)

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0,(1368,720))

#images_prev = ''
#for i,images in enumerate(names):
    #if i == 0:
        #images_prev = images
        #img_original = cv2.imread(images)
        #img = img_original
    #if images_prev == images:
        #img = img
    #else:
        #print(Folder + images_prev.split('/')[7] + '/keypoints/' + images_prev.split('/')[8])
        #cv2.imwrite(Folder + images_prev.split('/')[7] + '/keypoints/' + images_prev.split('/')[8] , img)
        ##cv2.imwrite(images_prev.split('/')[8] , img)
        ##res = cv2.bitwise_and(img_original,img_original,mask = img)
        ##cv2.imwrite('b_' + images_prev.split('/')[8] , img_original)
        
        #out.write(img)
        #img_original = cv2.imread(images)
        #img = img_original
        #images_prev = images
    #points_array = np.array(points[i].splitlines()[0].split(','))
    #points_arranged = points_array.reshape(int(len(points_array)/3),3)
    #kp = points_arranged[:,0:3]
    #kp = np.round(kp.astype(np.float)).astype(np.int)
    #kp[:,0] = bb[i][0] + kp[:,0]*(bb[i][2]/64)
    #kp[:,1] = bb[i][1] + kp[:,1]*(bb[i][3]/64)
    #kp[:,2] = np.round(points_arranged[:,2].astype(np.float)*100).astype(np.int)
    ##print(kp)
    ##kp[:,2] = points_arranged[:,2]
    
    #cv2.rectangle(img,(int(bb[i][0]),int(bb[i][1])),(int(bb[i][0] + bb[i][2]),int(bb[i][1]+bb[i][3])),(0,255,0))
    ##cv2.rectangle(img, (100,100),(200,200),(255,0,0))

    #drawCar(img,kp,bb)
    ##drawPerson(img,kp,bb)
    ##img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
    ##fig,ax = plt.subplots(1)

    ##ax.imshow(img)
    ##rect = patches.Rectangle((bb[i][0],bb[i][1]),bb[i][2],bb[i][3],linewidth=1,edgecolor='r',facecolor='none')
    ##plt.scatter(kp[:,0], kp[:,1])
    ##plt.scatter(,bb[i][1], img[bb[i][1],bb[i][0]])
    ##ax.add_patch(rect)
    ##plt.show()
    ##print(Folder + images_prev.split('/')[6] + '/keypoints/' + images_prev.split('/')[7] )
    
    
#out.release() 



#with open('/home/dinesh/CarCrash/data/Fifth/bb_person.txt') as f:
    #lines = f.readlines()
#names = []
#bb = []
#for line in lines:
    #names.append(line.split(' ')[0])
    #bb.append(np.array(line.split(' ')[1:5]).astype(np.float))

#with open('/home/dinesh/CarCrash/data/Fifth/keypoints_person.txt') as f:
    #keypoints = f.readlines()

#points = []
#for point in keypoints:
    #points.append(point)

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0,(1368,720))

#images_prev = ''
#for i,images in enumerate(names):
    #if i == 0:
        #images_prev = images
        #img_original = cv2.imread(Folder + images_prev.split('/')[7] + '/keypoints/' + images_prev.split('/')[8])
        #img = img_original
    #if images_prev == images:
        #img = img
    #else:
        #print(Folder + images_prev.split('/')[7] + '/keypoints/' + images_prev.split('/')[8])
        #cv2.imwrite(Folder + images_prev.split('/')[7] + '/keypoints/' + images_prev.split('/')[8] , img)
        
        #out.write(img)
        #img_original = cv2.imread(Folder + images.split('/')[7] + '/keypoints/' + images.split('/')[8])
        #img = img_original
        #images_prev = images
    
    #points_array = np.array(points[i].splitlines()[0].split(','))
    #points_arranged = points_array.reshape(int(len(points_array)/3),3)
    #kp = points_arranged[:,0:3]
    #kp = np.round(kp.astype(np.float)).astype(np.int)
    #kp[:,0] = bb[i][0] + kp[:,0]*(bb[i][2]/64)
    #kp[:,1] = bb[i][1] + kp[:,1]*(bb[i][3]/64)
    #kp[:,2] = np.round(points_arranged[:,2].astype(np.float)*100).astype(np.int)
    ##print(kp)
    ##kp[:,2] = points_arranged[:,2]
    
    #cv2.rectangle(img,(int(bb[i][0]),int(bb[i][1])),(int(bb[i][0] + bb[i][2]),int(bb[i][1]+bb[i][3])),(255,0,0))

    #drawPerson(img,kp,bb)

    
#out.release() 
        
	
