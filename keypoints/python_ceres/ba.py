
from ceres import *
import numpy as np

def func(ps):
    return np.sum(ps**2)

def grad(ps):
    return 2 * ps

def triangulate_with_ceres(K_1,K_2,RT_1,RT_2,point_3d,point_2d):
    k_1 = np.array(K_1).reshape(9)
    k_2 = np.array(K_2).reshape(9)
    rt_1 = np.concatenate((np.array(RT_1)[0:3,0:3].reshape(9),np.array(RT_1)[0:3,3]))
    rt_2 = np.concatenate((np.array(RT_2)[0:3,0:3].reshape(9),np.array(RT_2)[0:3,3]))
    #rt_1 = np.transpose(np.array(RT_1)).reshape(12)#[0:3,0:3].reshape(9),np.array(RT_1)[0:3,3]))
    #rt_2 = np.transpose(np.array(RT_2)).reshape(12)#rt_2 = np.concatenate((np.array(RT_1)[0:3,0:3].reshape(9),np.array(RT_1)[0:3,3]))

    x0 = np.array([0,2,1,2], dtype=np.double)
    x1 = np.concatenate((rt_1,rt_2))#np.array([0.9,0.02,0.04,0.03,0.9,0.05,0.05,0.07,0.93,2,10,200,0.95,0.025,0.045,0.035,0.95,0.055,0.055,0.075,0.56,25,105,2005], dtype=np.double)
    x2 = np.concatenate((k_1,k_2))#np.array([900,0.0,500,0.0,901,400,0.0,0.0,1.0,901,0.0,501,0.0,902,401,0.0,0.0,1.0], dtype=np.double)
    x3 = np.array(point_2d, dtype=np.double)
    x4 = np.array(point_3d, dtype=np.double)
    print(x0)
    print(x1)
    print(x2)
    print(x3)
    print(x4)
    output = ba_optimize(func,grad,x0,x1,x2,x3,x4)
    return np.array([[output.x],[output.y],[output.z],[1]])


def triangulate_nview_with_ceres(K_all,RT_all,point_3d,point_2d):
    rt_final = []
    for i,RT in enumerate(RT_all):
        #print(RT)
        rt = np.concatenate((np.array(RT)[0:3,0:3].reshape(9),np.array(RT)[0:3,3]))
        rt_final = np.concatenate((rt_final,rt))#np.array([0.9,0.02,0.04,0.03,0.9,0.05,0.05,0.07,0.93,2,10,200,0.95,0.025,0.045,0.035,0.95,0.055,0.055,0.075,0.56,25,105,2005], dtype=np.double)
    
    k_final = []
    for i,K in enumerate(K_all):
        k = np.array(K).reshape(9)
        k_final = np.concatenate((k_final,k))#np.array([0.9,0.02,0.04,0.03,0.9,0.05,0.05,0.07,0.93,2,10,200,0.95,0.025,0.045,0.035,0.95,0.055,0.055,0.075,0.56,25,105,2005], dtype=np.double)
    x0 = np.array([0,len(K_all),1,len(K_all)], dtype=np.double)
    x3 = np.array(point_2d, dtype=np.double)
    x4 = np.array(point_3d, dtype=np.double)
    output = ba_optimize(func,grad,x0,rt_final,k_final,x3,x4)
    return np.array([[output.x],[output.y],[output.z],[1]])



def triangulate_nview_with_ceres(K_all,RT_all,point_3d,point_2d):
    rt_final = []
    for i,RT in enumerate(RT_all):
        #print(RT)
        rt = np.concatenate((np.array(RT)[0:3,0:3].reshape(9),np.array(RT)[0:3,3]))
        rt_final = np.concatenate((rt_final,rt))#np.array([0.9,0.02,0.04,0.03,0.9,0.05,0.05,0.07,0.93,2,10,200,0.95,0.025,0.045,0.035,0.95,0.055,0.055,0.075,0.56,25,105,2005], dtype=np.double)
    
    k_final = []
    for i,K in enumerate(K_all):
        k = np.array(K).reshape(9)
        k_final = np.concatenate((k_final,k))#np.array([0.9,0.02,0.04,0.03,0.9,0.05,0.05,0.07,0.93,2,10,200,0.95,0.025,0.045,0.035,0.95,0.055,0.055,0.075,0.56,25,105,2005], dtype=np.double)
    x0 = np.array([0,len(K_all),1,len(K_all)], dtype=np.double)
    x3 = np.array(point_2d, dtype=np.double)
    x4 = np.array(point_3d, dtype=np.double)
    output = ba_optimize(func,grad,x0,rt_final,k_final,x3,x4)
    return np.array([[output.x],[output.y],[output.z],[1]])




def car_fit_nview_with_ceres(K_all,RT_all,point_3d,point_2d,index,scale,RT_transform):
    rt_final = []
    for i,RT in enumerate(RT_all):
        rt = np.concatenate((np.array(RT)[0:3,0:3].reshape(9),np.array(RT)[0:3,3]))
        rt_final = np.concatenate((rt_final,rt))#np.array([0.9,0.02,0.04,0.03,0.9,0.05,0.05,0.07,0.93,2,10,200,0.95,0.025,0.045,0.035,0.95,0.055,0.055,0.075,0.56,25,105,2005], dtype=np.double)
    
    k_final = []
    for i,K in enumerate(K_all):
        k = np.array(K).reshape(9)
        k_final = np.concatenate((k_final,k))#np.array([0.9,0.02,0.04,0.03,0.9,0.05,0.05,0.07,0.93,2,10,200,0.95,0.025,0.045,0.035,0.95,0.055,0.055,0.075,0.56,25,105,2005], dtype=np.double)
    x0 = np.array([0,len(K_all),len(point_3d)/3,len(K_all)], dtype=np.double)
    x3 = np.array(point_2d, dtype=np.double)
    x4 = np.array(point_3d, dtype=np.double)
    x5 = np.array(index, dtype=np.double)
    x6 = np.array([scale], dtype=np.double)
    RT_transform = np.array(RT_transform)
    a = np.concatenate((RT_transform[0:3,0:3].reshape(9),RT_transform[0:3,3]))
    x7 = np.array(a, dtype=np.double)
    #print(rt_final[0])
    #print(x5)
    output = ba_optimize(func,grad,x0,rt_final,k_final,x3,x4,x5,x6,x7)
    #RT = np.array([[output.rt00,output.rt01,output.rt02,output.rt03],[output.rt10,output.rt11,output.rt12,output.rt13],[output.rt20,output.rt21,output.rt22,output.rt23],[0,0,0,1]])
    carrt_all = output.rt
    lst = carrt_all.split(',')
    carrt_float = [float(i) for i in lst]
    RT = np.array([[carrt_float[0],carrt_float[1],carrt_float[2],carrt_float[9]],[carrt_float[3],carrt_float[4],carrt_float[5],carrt_float[10]],[carrt_float[6],carrt_float[7],carrt_float[8],carrt_float[11]],[0,0,0,1]])
    return output.scale,RT

#np.array([[output.rt00,output.rt01,output.rt02,output.rt03],[output.rt10,output.rt11,output.rt12,output.rt13],[output.rt20,output.rt21,output.rt22,output.rt23],[0,0,0,1]])
#K_1 = [[1417.168739, 0.0, 976.65509499999996], [0.0, 1419.4286790000001, 553.24186499999996], [0.0, 0.0, 1.0]]
#K_2 = [[1401.7311609999999, 0.0, 953.57361500000002], [0.0, 1382.306507, 551.61246100000005], [0.0, 0.0, 1.0]]
#RT_1 = [[ 0.68753503, -0.00441855, -0.72613777,  2.76380242],
#       [ 0.04099636,  0.99862274,  0.0327403 ,  0.16624821],
#       [ 0.72499302, -0.05227911,  0.68676926,  2.62888346]]
#RT_2 = [[ 0.77938564,  0.00471068,  0.6265268 ,  1.28058797],
#       [ 0.00676228,  0.99985025, -0.01592972, -0.02051908],
#       [-0.62650801,  0.01665214,  0.77923707,  0.26322107]]
#point_3d = [-2.39330017,0.13540689,1.86683976]
#point_2d = [1443.3709677419354, 777.95161290322574,931.49428360685943, 600.76761676457397]
#triangulate_with_ceres(K_1,K_2,RT_1,RT_2,point_3d,point_2d)

def car_fit_video_nview_with_ceres(K_all, RT_all, point_3d, point_2d, index, scale, RT_transform,car_time):
    k_final = []
    for i,K in enumerate(K_all):
        k = np.array(K).reshape(9)
        k_final = np.concatenate((k_final,k))#np.array([0.9,0.02,0.04,0.03,0.9,0.05,0.05,0.07,0.93,2,10,200,0.95,0.025,0.045,0.035,0.95,0.055,0.055,0.075,0.56,25,105,2005], dtype=np.double)

    rt_final = []
    for i,RT in enumerate(RT_all):
        rt = np.concatenate((np.array(RT)[0:3,0:3].reshape(9),np.array(RT)[0:3,3]))
        rt_final = np.concatenate((rt_final,rt))#np.array([0.9,0.02,0.04,0.03,0.9,0.05,0.05,0.07,0.93,2,10,200,0.95,0.025,0.045,0.035,0.95,0.055,0.055,0.075,0.56,25,105,2005], dtype=np.double)
    
    car_rt_final = []
    for j,car_RT in enumerate(RT_transform):
        car_rt = np.concatenate((np.array(car_RT)[0:3,0:3].reshape(9),np.array(car_RT)[0:3,3]))
        car_rt_final = np.concatenate((car_rt_final,car_rt))#np.array([0.9,0.02,0.04,0.03,0.9,0.05,0.05,0.07,0.93,2,10,200,0.95,0.025,0.045,0.035,0.95,0.055,0.055,0.075,0.56,25,105,2005], dtype=np.double)
    

    x0 = np.array([0,len(RT_transform),len(point_3d)/3,len(K_all)], dtype=np.double)
    x3 = np.array(point_2d, dtype=np.double)
    x4 = np.array(point_3d, dtype=np.double)
    x5 = np.array(index, dtype=np.double)
    x6 = np.array([scale], dtype=np.double)
    x7 = np.array(car_time, dtype=np.double)
#    RT_transform = np.array(RT_transform)
#    a = np.concatenate((RT_transform[0:3,0:3].reshape(9),RT_transform[0:3,3]))
#    x7 = np.array(a, dtype=np.double)
    output = ba_optimize_video(func,grad,x0,rt_final,k_final,x3,x4,x5,x6,car_rt_final,x7)
    #RT = np.array([[output.rt00,output.rt01,output.rt02,output.rt03],[output.rt10,output.rt11,output.rt12,output.rt13],[output.rt20,output.rt21,output.rt22,output.rt23],[0,0,0,1]])
    carrt_all = output.rt
    lst = carrt_all.split(',')
    carrt_float_all = [float(i) for i in lst]
    RT = []
    for j,car_RT in enumerate(RT_transform): 
        carrt_float = carrt_float_all[j*12:j*12+12]
        RT.append(np.array([[carrt_float[0],carrt_float[1],carrt_float[2],carrt_float[9]],[carrt_float[3],carrt_float[4],carrt_float[5],carrt_float[10]],[carrt_float[6],carrt_float[7],carrt_float[8],carrt_float[11]],[0,0,0,1]]))
        #print(RT)
    return output.scale,RT
