import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches





Folder = '/home/dinesh/CarCrash/data/Fifth/'

def drawCar(img,keypoints,bb):
#    if  keypoints[0,2] >50 and keypoints[2,2] >50:
#        cv2.line(img,tuple(keypoints[0,0:2]),tuple(keypoints[2,0:2]),(255,0,0),5)
#    if  keypoints[1,2] >50 and keypoints[3,2] >50:
#        cv2.line(img,tuple(keypoints[1,0:2]),tuple(keypoints[3,0:2]),(0,0,255),5)
#    if  keypoints[0,2] >50 and keypoints[1,2] >50:
#        cv2.line(img,tuple(keypoints[0,0:2]),tuple(keypoints[1,0:2]),(0,255,0),5)
#    if  keypoints[3,2] >50 and keypoints[2,2] >50:
#        cv2.line(img,tuple(keypoints[3,0:2]),tuple(keypoints[2,0:2]),(128,128,0),5)

    # wheels
    if  keypoints[1,2] >5:
        cv2.circle(img,tuple(keypoints[1,0:2]),3,(0,255,0),3)
    if  keypoints[2,2] >5:
        cv2.circle(img,tuple(keypoints[2,0:2]),3,(64,255,255),3)
    if  keypoints[3,2] >5:
        	cv2.circle(img,tuple(keypoints[3,0:2]),3,(128,255,128),3)
    if  keypoints[0,2] >5:
        	cv2.circle(img,tuple(keypoints[0,0:2]),3,(128,255,0),3)

    if  keypoints[0,2] >5 and keypoints[2,2] >5:
        cv2.line(img,tuple(keypoints[0,0:2]),tuple(keypoints[2,0:2]),(255,0,0),2)
    if  keypoints[1,2] >5 and keypoints[3,2] >5:
        cv2.line(img,tuple(keypoints[1,0:2]),tuple(keypoints[3,0:2]),(255,0,0),2)
    if  keypoints[0,2] >5 and keypoints[1,2] >5:
        cv2.line(img,tuple(keypoints[0,0:2]),tuple(keypoints[1,0:2]),(255,0,0),2)
    if  keypoints[2,2] >5 and keypoints[3,2] >5:
        cv2.line(img,tuple(keypoints[2,0:2]),tuple(keypoints[3,0:2]),(255,0,0),2)



    # top of car
    if  keypoints[10,2] >5:
        	cv2.circle(img,tuple(keypoints[10,0:2]),3,(255,128,128),3)
    if  keypoints[11,2] >5:
        	cv2.circle(img,tuple(keypoints[11,0:2]),3,(128,128,128),3)
    if  keypoints[12,2] >5:
        	cv2.circle(img,tuple(keypoints[12,0:2]),3,(0,128,255),3)
    if  keypoints[13,2] >5:
        	cv2.circle(img,tuple(keypoints[13,0:2]),3,(0,255,255),3)

    if  keypoints[10,2] >5 and keypoints[12,2] >5:
        cv2.line(img,tuple(keypoints[10,0:2]),tuple(keypoints[12,0:2]),(0,255,0),2)
    if  keypoints[11,2] >5 and keypoints[13,2] >5:
        cv2.line(img,tuple(keypoints[11,0:2]),tuple(keypoints[13,0:2]),(0,255,0),2)
    if  keypoints[10,2] >5 and keypoints[11,2] >5:
        cv2.line(img,tuple(keypoints[10,0:2]),tuple(keypoints[11,0:2]),(0,255,0),2)
    if  keypoints[12,2] >5 and keypoints[13,2] >5:
        cv2.line(img,tuple(keypoints[12,0:2]),tuple(keypoints[13,0:2]),(0,255,0),2)
        
    # front head lights
    if  keypoints[4,2] >20:
        	cv2.circle(img,tuple(keypoints[4,0:2]),3,(0,255,0),3)
    if  keypoints[0,2] >5 and keypoints[4,2] >20:
        cv2.line(img,tuple(keypoints[0,0:2]),tuple(keypoints[4,0:2]),(0,0,255),2)
    if  keypoints[10,2] >5 and keypoints[4,2] >20:
        cv2.line(img,tuple(keypoints[10,0:2]),tuple(keypoints[4,0:2]),(0,0,255),2)

    if  keypoints[5,2] >20:
        cv2.circle(img,tuple(keypoints[5,0:2]),3,(128,0,0),3)
    if  keypoints[1,2] >5 and keypoints[5,2] >20:
        cv2.line(img,tuple(keypoints[1,0:2]),tuple(keypoints[5,0:2]),(0,0,255),2)
    if  keypoints[11,2] >5 and keypoints[5,2] >20:
        cv2.line(img,tuple(keypoints[11,0:2]),tuple(keypoints[5,0:2]),(0,0,255),2)
    if  keypoints[4,2] >20 and keypoints[5,2] >20:
        cv2.line(img,tuple(keypoints[4,0:2]),tuple(keypoints[5,0:2]),(0,0,255),2)

    # back head lights
    if  keypoints[6,2] >20:
        	cv2.circle(img,tuple(keypoints[6,0:2]),3,(255,0,0),3)
    if  keypoints[2,2] >5 and keypoints[6,2] >20:
        cv2.line(img,tuple(keypoints[2,0:2]),tuple(keypoints[6,0:2]),(255,0,255),2)
    if  keypoints[12,2] >5 and keypoints[6,2] >20:
        cv2.line(img,tuple(keypoints[12,0:2]),tuple(keypoints[6,0:2]),(255,0,255),2)

    
    if  keypoints[7,2] >20:
        	cv2.circle(img,tuple(keypoints[7,0:2]),3,(255,0,128),5)
    if  keypoints[3,2] >5 and keypoints[7,2] >20:
        cv2.line(img,tuple(keypoints[3,0:2]),tuple(keypoints[7,0:2]),(255,0,255),2)
    if  keypoints[13,2] >5 and keypoints[7,2] >20:
        cv2.line(img,tuple(keypoints[13,0:2]),tuple(keypoints[7,0:2]),(255,0,255),2)
    if  keypoints[6,2] >20 and keypoints[7,2] >20:
        cv2.line(img,tuple(keypoints[6,0:2]),tuple(keypoints[7,0:2]),(255,0,255),2)

    # mirrror
    if  keypoints[8,2] >20:
        	cv2.circle(img,tuple(keypoints[8,0:2]),3,(128,0,128),5)
    if  keypoints[8,2] >20 and keypoints[4,2] >20:
        cv2.line(img,tuple(keypoints[8,0:2]),tuple(keypoints[4,0:2]),(0,0,255),2)
    
    if  keypoints[9,2] >20:
        cv2.circle(img,tuple(keypoints[9,0:2]),3,(0,128,128),5)
    if  keypoints[9,2] >20 and keypoints[5,2] >20:
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

def drawPerson(img,keypoints,bb):

    # wheels
    if  keypoints[0,2] >0:
        	cv2.circle(img,tuple(keypoints[0,0:2]),3,(128,255,0),2)
    if  keypoints[1,2] >0:
        cv2.circle(img,tuple(keypoints[1,0:2]),3,(0,255,0),2)
    if  keypoints[2,2] >0:
        cv2.circle(img,tuple(keypoints[2,0:2]),3,(64,255,255),2)
    if  keypoints[3,2] >0:
        	cv2.circle(img,tuple(keypoints[3,0:2]),3,(128,255,128),2)
    if  keypoints[4,2] >0:
        	cv2.circle(img,tuple(keypoints[4,0:2]),3,(0,255,0),2)
    if  keypoints[5,2] >0:
        cv2.circle(img,tuple(keypoints[5,0:2]),3,(128,0,0),2)
    if  keypoints[6,2] >0:
        	cv2.circle(img,tuple(keypoints[6,0:2]),3,(255,0,0),2)
    if  keypoints[7,2] >0:
        	cv2.circle(img,tuple(keypoints[7,0:2]),3,(255,0,128),2)
    if  keypoints[8,2] >0:
        	cv2.circle(img,tuple(keypoints[8,0:2]),3,(128,0,128),2)
    if  keypoints[9,2] >0:
        cv2.circle(img,tuple(keypoints[9,0:2]),3,(0,128,128),2)
    if  keypoints[10,2] >0:
        	cv2.circle(img,tuple(keypoints[10,0:2]),3,(255,128,128),2)
    if  keypoints[11,2] >0:
        	cv2.circle(img,tuple(keypoints[11,0:2]),3,(128,128,128),2)
    if  keypoints[12,2] >0:
        	cv2.circle(img,tuple(keypoints[12,0:2]),3,(0,128,255),2)
    if  keypoints[13,2] >0:
        	cv2.circle(img,tuple(keypoints[13,0:2]),3,(0,255,255),2)
    if  keypoints[14,2] >0:
        	cv2.circle(img,tuple(keypoints[14,0:2]),3,(0,255,64),2)
    if  keypoints[15,2] >0:
        	cv2.circle(img,tuple(keypoints[15,0:2]),3,(0,64,255),2)

    if  keypoints[0,2] >0 and keypoints[1,2] >0:
        cv2.line(img,tuple(keypoints[0,0:2]),tuple(keypoints[1,0:2]),(0,255,0),2)
    if  keypoints[1,2] >0 and keypoints[2,2] >0:
        cv2.line(img,tuple(keypoints[1,0:2]),tuple(keypoints[2,0:2]),(0,255,0),2)
    if  keypoints[2,2] >0 and keypoints[6,2] >0:
        cv2.line(img,tuple(keypoints[2,0:2]),tuple(keypoints[6,0:2]),(0,255,0),2)
    if  keypoints[3,2] >0 and keypoints[4,2] >0:
        cv2.line(img,tuple(keypoints[3,0:2]),tuple(keypoints[4,0:2]),(0,255,0),2)
    if  keypoints[3,2] >0 and keypoints[6,2] >0:
        cv2.line(img,tuple(keypoints[3,0:2]),tuple(keypoints[6,0:2]),(0,255,0),2)
    if  keypoints[4,2] >0 and keypoints[5,2] >0:
        cv2.line(img,tuple(keypoints[4,0:2]),tuple(keypoints[5,0:2]),(0,255,0),2)
    if  keypoints[6,2] >0 and keypoints[8,2] >0:
        cv2.line(img,tuple(keypoints[6,0:2]),tuple(keypoints[8,0:2]),(0,255,0),2)
    if  keypoints[8,2] >0 and keypoints[9,2] >0:
        cv2.line(img,tuple(keypoints[8,0:2]),tuple(keypoints[9,0:2]),(0,255,0),2)
    if  keypoints[13,2] >0 and keypoints[8,2] >0:
        cv2.line(img,tuple(keypoints[13,0:2]),tuple(keypoints[8,0:2]),(0,255,0),2)
    if  keypoints[10,2] >0 and keypoints[11,2] >0:
        cv2.line(img,tuple(keypoints[10,0:2]),tuple(keypoints[11,0:2]),(0,255,0),2)
    if  keypoints[11,2] >0 and keypoints[12,2] >0:
        cv2.line(img,tuple(keypoints[11,0:2]),tuple(keypoints[12,0:2]),(0,255,0),2)
    if  keypoints[12,2] >0 and keypoints[8,2] >0:
        cv2.line(img,tuple(keypoints[12,0:2]),tuple(keypoints[8,0:2]),(0,255,0),2)
    if  keypoints[13,2] >0 and keypoints[14,2] >0:
        cv2.line(img,tuple(keypoints[13,0:2]),tuple(keypoints[14,0:2]),(0,255,0),2)
    if  keypoints[14,2] >0 and keypoints[15,2] >0:
        cv2.line(img,tuple(keypoints[14,0:2]),tuple(keypoints[15,0:2]),(0,255,0),2)
        
    # front head lights
    


with open('/home/dinesh/CarCrash/data/Fifth/bb_car.txt') as f:
#with open('/home/dinesh/CarCrash/data/Fifth/bb_person.txt') as f:
    lines = f.readlines()
names = []
bb = []
for line in lines:
    #print(line)
    names.append(line.split(' ')[0])
    bb.append(np.array(line.split(' ')[1:5]).astype(np.float))

with open('/home/dinesh/CarCrash/data/Fifth/keypoints_car.txt') as f:
#with open('/home/dinesh/CarCrash/data/Fifth/keypoints_person.txt') as f:
    keypoints = f.readlines()

points = []
for point in keypoints:
    #print(line)
    points.append(point)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0,(1368,720))

images_prev = ''
for i,images in enumerate(names):
    if i == 0:
        images_prev = images
        img_original = cv2.imread(images)
        img = img_original
    if images_prev == images:
        img = img
    else:
        print(Folder + images_prev.split('/')[7] + '/keypoints/' + images_prev.split('/')[8])
        cv2.imwrite(Folder + images_prev.split('/')[7] + '/keypoints/' + images_prev.split('/')[8] , img)
        #cv2.imwrite(images_prev.split('/')[8] , img)
        #res = cv2.bitwise_and(img_original,img_original,mask = img)
        #cv2.imwrite('b_' + images_prev.split('/')[8] , img_original)
        
        out.write(img)
        img_original = cv2.imread(images)
        img = img_original
        images_prev = images
    if i == 2000:    
   		break
    points_array = np.array(points[i].splitlines()[0].split(','))
    points_arranged = points_array.reshape(int(len(points_array)/3),3)
    kp = points_arranged[:,0:3]
    kp = np.round(kp.astype(np.float)).astype(np.int)
    kp[:,0] = bb[i][0] + kp[:,0]*(bb[i][2]/64)
    kp[:,1] = bb[i][1] + kp[:,1]*(bb[i][3]/64)
    kp[:,2] = np.round(points_arranged[:,2].astype(np.float)*100).astype(np.int)
    #print(kp)
    #kp[:,2] = points_arranged[:,2]
    
    cv2.rectangle(img,(int(bb[i][0]),int(bb[i][1])),(int(bb[i][0] + bb[i][2]),int(bb[i][1]+bb[i][3])),(0,255,0))
    #cv2.rectangle(img, (100,100),(200,200),(255,0,0))

    drawCar(img,kp,bb)
    #drawPerson(img,kp,bb)
    #img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
    #fig,ax = plt.subplots(1)

    #ax.imshow(img)
    #rect = patches.Rectangle((bb[i][0],bb[i][1]),bb[i][2],bb[i][3],linewidth=1,edgecolor='r',facecolor='none')
    #plt.scatter(kp[:,0], kp[:,1])
    #plt.scatter(,bb[i][1], img[bb[i][1],bb[i][0]])
    #ax.add_patch(rect)
    #plt.show()
    #print(Folder + images_prev.split('/')[6] + '/keypoints/' + images_prev.split('/')[7] )
    
    
out.release() 



with open('/home/dinesh/CarCrash/data/Fifth/bb_person.txt') as f:
    lines = f.readlines()
names = []
bb = []
for line in lines:
    names.append(line.split(' ')[0])
    bb.append(np.array(line.split(' ')[1:5]).astype(np.float))

with open('/home/dinesh/CarCrash/data/Fifth/keypoints_person.txt') as f:
    keypoints = f.readlines()

points = []
for point in keypoints:
    points.append(point)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0,(1368,720))

images_prev = ''
for i,images in enumerate(names):
    if i == 0:
        images_prev = images
        img_original = cv2.imread(Folder + images_prev.split('/')[7] + '/keypoints/' + images_prev.split('/')[8])
        img = img_original
    if images_prev == images:
        img = img
    else:
        print(Folder + images_prev.split('/')[7] + '/keypoints/' + images_prev.split('/')[8])
        cv2.imwrite(Folder + images_prev.split('/')[7] + '/keypoints/' + images_prev.split('/')[8] , img)
        cv2.imwrite('b_' + images_prev.split('/')[8] , img_original)
        
        out.write(img)
        img_original = cv2.imread(Folder + images_prev.split('/')[7] + '/keypoints/b_' + images_prev.split('/')[8])
        img = img_original
        images_prev = images
    
    if i == 2000:    
   		break    
   		
    points_array = np.array(points[i].splitlines()[0].split(','))
    points_arranged = points_array.reshape(int(len(points_array)/3),3)
    kp = points_arranged[:,0:3]
    kp = np.round(kp.astype(np.float)).astype(np.int)
    kp[:,0] = bb[i][0] + kp[:,0]*(bb[i][2]/64)
    kp[:,1] = bb[i][1] + kp[:,1]*(bb[i][3]/64)
    kp[:,2] = np.round(points_arranged[:,2].astype(np.float)*100).astype(np.int)
    #print(kp)
    #kp[:,2] = points_arranged[:,2]
    
    cv2.rectangle(img,(int(bb[i][0]),int(bb[i][1])),(int(bb[i][0] + bb[i][2]),int(bb[i][1]+bb[i][3])),(0,255,0))

    drawPerson(img,kp,bb)

    
out.release() 
        
	
