import numpy as np
import cv2




def time_instance_data(data,synched_images,RT,RT_index,K,K_index):
    all_bb = []
    cam_index = []
    RT_all = []
    kp_all = []
    K_all = []
    for cam,num in enumerate(synched_images):
        Index_num = np.where(np.array(RT_index[cam]).astype(np.int) == num)
        if len(Index_num[0]) == 1:
                RT_cam = RT[cam][int(Index_num[0])]
                K_cam = K[cam][int(Index_num[0])]
                if len(data[cam]) > 1: #try:
                    for index,keypoints in enumerate(data[cam]):
                        bb =np.array([keypoints[1],keypoints[2],keypoints[3],keypoints[4]]).astype(np.int)

                        points_array = np.array(keypoints[5:])
                        points_arranged = points_array.reshape(14,3)
                        kp = np.round(points_arranged.astype(np.float)).astype(np.int)
                        kp[:,0] = bb[0] + kp[:,0]*(bb[2]/64)
                        kp[:,1] = bb[1] + kp[:,1]*(bb[3]/64)
                        kp[:,2] = np.round(points_arranged[:,2].astype(np.float)*100).astype(np.int)
                        #print(np.sqrt(bb[2]**2+bb[3]**2))
                        if np.sqrt(bb[2]**2+bb[3]**2) > 100 and np.sqrt(bb[2]**2+bb[3]**2) < 1000 and cam != 6 and cam != 8:
                            all_bb.append(bb)
                            kp_all.append(kp)
                            cam_index.append(cam)
                            RT_all.append(RT_cam)
                            K_all.append(K_cam)
    return all_bb,kp_all,cam_index,RT_all,K_all

def read_extrinsics(filename):
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

def read_intrinsics(filename):
    with open(filename) as f:
        lines = f.readlines()
    index = []
    K = []
    for line in lines:
        index.append(line.split(' ')[0])
        Int = np.array(line.split(' ')[5:10]).astype(np.float)
        K_computed = [[Int[0],0.0,Int[3]],[0.0,Int[1],Int[4]],[0.0,0.0,1.0]]
        #print(Int)
        K.append(K_computed)
    return K, index

def Read_keypoints(synched_images,Folder):
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

def extrinsics(Folder):
    RT = []
    RT_index = []
    for cam in range(21):
        try:
            RT_cam, index_cam = read_extrinsics(Folder +  '/vHCamPose_RSCayley_' + str(cam) + '.txt')
        except:
            RT_cam = []
            index_cam = []#read_extrinsics('/home/dinesh/CarCrash/data/Fifth/avCamPose_RSCayley_'+ str(cam) + '.txt')
        RT.append(RT_cam)#[])
        RT_index.append(index_cam)#[])
        print('reading extrinsics of ' , cam)
    return RT,RT_index

def intrinsics(Folder):
    K = []
    K_index = []
    for cam in range(21):
        try:
            K_cam, index_K_cam = read_intrinsics(Folder + '/vHIntrinsic_'+ str(cam) + '.txt')
            #print(cam)
        except:
            K_cam = []
            index_K_cam = []#read_func('/home/dinesh/CarCrash/data/Fifth/avCamPose_RSCayley_'+ str(cam) + '.txt')
        K.append(K_cam)#[])
        K_index.append(index_K_cam)#[])
        print('reading intrinsics of ' , cam)
    return K,K_index
    
    
def cameraSync(Folder):
    with open(Folder + '/InitSync.txt') as f:
        lines = f.readlines()
    index = []
    diff = []
    for line in lines:
        index.append(line.split('\t')[0])
        diff.append(line.split('\t')[1].split('\r')[0])
    
    diff = np.array(diff).astype(int)
    return diff

