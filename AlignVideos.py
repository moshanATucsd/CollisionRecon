import cv2
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

import cv2
import os
#from align_audio.alignment_by_row_channels import align


def processInput(i,VideoFile,folder_path,output_path,time):

	#bgs = libbgs.PixelBasedAdaptiveSegmenter()
	video_file = os.path.join(folder_path,'RAW',str(i),VideoFile)
	capture = cv2.VideoCapture(video_file)
	time_path1 = folder_path + '/Cleaned2/' + str(i) + '/time.txt'
	directory1 = folder_path + '/RAW/' + str(0) +'/'
	directory2 = folder_path + '/RAW/' + str(i) + '/'
	time_path = output_path + str(i) + '/time.txt'
	if os.path.isfile(time_path):
		t = np.loadtxt(time_path)
	else:
		t = np.loadtxt(time_path1)
		#t = align(VideoFile, VideoFile, directory1, directory2) 
		t_save = [t]
		np.savetxt(time_path,t_save)
	time[i] = time[0] + t*1000
	width = int(capture.get(3))
	height =int(capture.get(4))
	fps = 120
	print(width,height,fps)
	fourcc = cv2.cv.CV_FOURCC(*'XVID')
	#print(output_path + str(i) + '/output.avi')
	out = cv2.VideoWriter(output_path + str(i) + '/x.avi',fourcc,fps,(width,height))
	#segmented = cv2.VideoWriter(output_path + str(i) + '/segmented.avi',fourcc,fps,(width,height))
	capture.set(0,time[i])
	pos_frame_intial = capture.get(0)
	print(pos_frame_intial)
	num = 1
	if not os.path.exists(output_path + str(i) + '/images/'):
                os.makedirs(output_path + str(i) + '/images/')

	while num < 1000:
		flag, frame = capture.read()
		if flag:
			pos_frame = capture.get(1)
			#cv2.imwrite(output_path + str(i) + '/images/' + str(num).zfill(4) + '.png', frame)
			#img_output = BackGroundSub(frame,bgs)
                        #print((output_path + str(i) + '/images/' + str(num).zfill(4) + '.png'))
                        out.write(frame)
			#segmented.write(img_output)
			#cv2.imwrite(output_path + str(i) + '/masks/' + str(num).zfill(4) + '.jpg.png', img_output)
			#cv2.imshow('img_bgmodel', img_bgmodel)
		else:
			capture.set(1, pos_frame_intial-1)
			print("Frame is not ready")
			#cv2.waitKey(1)
			# break
		num = num+1
	#cv2.destroyAllWindows()


folder_path = '/home/dinesh/CarCrash/data/CarCrash/'
output_path = folder_path + 'Crash2/'
inputs = [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,20,21,24,25,26] #all sequence2
#inputs = [11,12,13,14,15,16,17,18,21,25,26]
inputs = [21,22,23,25,26]
time = np.zeros(30)
time[0] = 201000
#time[10] = 371683 # sequence 2
#time[0] = 342000 #sequence 1
videofile = 'IMG_0002.MOV'
num_cores = multiprocessing.cpu_count()-2
results = Parallel(n_jobs=num_cores)(delayed(processInput)(i,videofile,folder_path,output_path,time) for i in inputs)
