def Reproject_error(RT_all,K_all,all_bb,cam_index,location_3d):
    final_error = 0
    cam_num = 0
    center_boundingbox_list = []
    for i,boundingbox in enumerate(all_bb):
        P = np.dot(K_all[i],  RT_all[i])
        point_2d = np.dot(P,location_3d)
        point_2d = point_2d/np.tile(point_2d[-1, :], (3, 1))
        center_boundingbox = np.array([[boundingbox[0]+boundingbox[2]/2],[boundingbox[1]+boundingbox[3]/2]])
        size = boundingbox[3]**2 + boundingbox[2]**2
        if point_2d[0] > boundingbox[0] and point_2d[0] < boundingbox[0] + boundingbox[2] and point_2d[1] > boundingbox[1] and point_2d[1] < boundingbox[1] + boundingbox[3]:
            final_error += np.sqrt(np.mean((point_2d[0:-1] - center_boundingbox)**2))/np.sqrt(size)
    
    if final_error == 0:
        for i,boundingbox in enumerate(all_bb):
            P = np.dot(K_all[i],  RT_all[i])
            point_2d = np.dot(P,location_3d)
            point_2d = point_2d/np.tile(point_2d[-1, :], (3, 1))
            center_boundingbox = np.array([[boundingbox[0]+boundingbox[2]/2],[boundingbox[1]+boundingbox[3]/2]])
            center_boundingbox_list.append(center_boundingbox)
            if cam_num != cam_index[i]:
                #print(np.min(np.mean((center_boundingbox_list - point_2d[0:-1])**2)))
                final_error += np.sqrt(np.min(np.mean((center_boundingbox_list - point_2d[0:-1])**2)))
                cam_num = cam_index[i]
                center_boundingbox_list = []
    return final_error



def Draw_camera(synched_images,data, RT, RT_index):
    for cam,num in enumerate(synched_images):
        Index_num = np.where(np.array(RT_index[cam]).astype(np.int) == num)
        if len(Index_num[0]) == 1:
            RT_1 = RT[cam][int(Index_num[0])]
            print(RT_1)
        else:
            continue


def loop(synched_images,RT_1,bb,corr):
    for cam_comp,num_comp in enumerate(synched_images):
        if len(data[cam_comp]) > 1:
            for index_comp,keypoints_comp in enumerate(data[cam_comp]):
                bb_compare = np.array([keypoints_comp[1],keypoints_comp[2],keypoints_comp[3],keypoints_comp[4]]).astype(np.int)
                Index_num_compare = np.where(np.array(RT_index[cam_comp]).astype(np.int) == num_comp)
                if len(Index_num_compare[0]) == 1 and cam != cam_comp:
                    RT_compare = RT[cam_comp][int(Index_num_compare[0])]
                    center_bb = np.array([[bb_compare[0]+bb_compare[2]/2],[bb_compare[1]+bb_compare[3]/2]])
                    center_bb_compare = np.array([[bb_compare[0]+bb_compare[2]/2],[bb_compare[1]+bb_compare[3]/2]])
                    P_1 = np.dot(K,  RT_1) 
                    P_2 = np.dot(K,  RT_compare)
                    #print(RT_1)
                    point_4d_hom = cv2.triangulatePoints(P_1,P_2,center_bb,center_bb_compare)
                    #print(point_4d_hom)
                    #print('check')
                    point_4d_hom = point_4d_hom.astype(np.float)
                    point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
                    #print(point_4d_hom)
                    #point_4d = [[-3.2419],[0.0352],[1.0953],[1]];
                    #print(point_4d)
                    #print(RT_1)
                    point_2d_1 = np.dot(P_1,point_4d)
                    point_2d_1 = point_2d_1/np.tile(point_2d_1[-1, :], (3, 1))
                    #print(point_2d_1)

                    point_2d_2 = np.dot(P_2,point_4d)
                    point_2d_2 = point_2d_1/np.tile(point_2d_1[-1, :], (3, 1))
                    
                    if np.sqrt(np.mean((point_2d_1[0:-1] - center_bb)**2)) + np.sqrt(np.mean((point_2d_1[0:-1] - center_bb)**2)) < 1000:
                        print(cam_comp,int(Index_num_compare[0]),bb_compare)
                        print(point_2d_1,center_bb)
                    #if cam_comp == 0:
                    #    print(cam_comp)
                    #    print(int(Index_num_compare[0]))
                    #    print(RT_compare)
                    #print(np.sqrt(np.mean((point_2d_1[0:-1] - center_bb)**2)) + np.sqrt(np.mean((point_2d_2[0:-1] - center_bb_compare)**2)))
                    #print(np.sqrt(np.mean((point_2d_2[0:-1] - center_bb_compare)**2)))
                    #print('done')
                    #point_4d = point_4d[:3, :].T
                else:
                    continue
        else:
            corr.append([])

def correspondense(synched_images,data, RT, RT_index):
    corr = []
    for cam,num in enumerate(synched_images):
        Index_num = np.where(np.array(RT_index[cam]).astype(np.int) == num)
        if len(Index_num[0]) == 1:
            RT_1 = RT[cam][int(Index_num[0])]
            if len(data[cam]) > 1: #try:
                for index,keypoints in enumerate(data[cam]):
                    bb = np.array([keypoints[1],keypoints[2],keypoints[3],keypoints[4]]).astype(np.int)
                    print(cam,num,bb)
                    loop(synched_images,RT_1,bb,corr)
                    print('done')
            else:
                corr.append([])
        else:
            corr.append([])
    return corr

def ransac_correspondense(synched_images,data, RT, RT_index):
    corr = []
    ## find all boundix boxes
    all_bb = []
    cam_index = []
    RT_all = []
    for cam,num in enumerate(synched_images):
        Index_num = np.where(np.array(RT_index[cam]).astype(np.int) == num)
        if len(Index_num[0]) == 1:
                RT_cam = RT[cam][int(Index_num[0])]
                if len(data[cam]) > 1: #try:
                    for index,keypoints in enumerate(data[cam]):
                        all_bb.append(np.array([keypoints[1],keypoints[2],keypoints[3],keypoints[4]]).astype(np.int))
                        cam_index.append(cam)
                        RT_all.append(RT_cam)


    
    for cam,num in enumerate(synched_images):
        Index_num = np.where(np.array(RT_index[cam]).astype(np.int) == num)
        if len(Index_num[0]) == 1:
            RT_1 = RT[cam][int(Index_num[0])]
            if len(data[cam]) > 1: #try:
                for index,keypoints in enumerate(data[cam]):
                    bb = np.array([keypoints[1],keypoints[2],keypoints[3],keypoints[4]]).astype(np.int)
                    print(cam,num,bb)
                    loop(synched_images,RT_1,bb,corr)
                    print('done')
            else:
                corr.append([])
        else:
            corr.append([])
    return corr
