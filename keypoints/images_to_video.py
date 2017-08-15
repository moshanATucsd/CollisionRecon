import numpy as np
from PIL import Image
import cv2
folder_path = '/home/dinesh/CarCrash/data/Fifth/'

filepath = folder_path + '/InitSync.txt'
with open(filepath) as f:
	lines = f.readlines()
index = []
diff = []
for line in lines:
	index.append(line.split('\t')[0])
	diff.append(line.split('\t')[1].split('\r')[0])

diff = np.array(diff).astype(int)
index = np.array(index).astype(int)

scale = 0.2
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('final.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60.0, (int(7680*scale),int(4320*scale)))

for time in range(9500):
	#time =4000
	diff_new = time - diff
	list_im = []
	print(time)
	for num,value in enumerate(index):
		filename = folder_path + str(value) + '/labelled_image/' + str(diff_new[num]).zfill(5) + '.png'
		if value == 6 or value == 10 or value == 16:
			continue
		list_im.append(filename)

	imgs = []
	for i in list_im:
		try:
			h = Image.open(i)
			h = h.resize( [int(scale * s) for s in h.size])
		except:
			h = Image.new('RGB', (int(7680*scale/4),int(4320*scale/4)))
		imgs.append(h)
	min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]

	imgs_v = []
	for v in range(4):
		imgs_new = imgs[v*4:v*4 + 4]
		imgs_comb_v = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs_new))
		imgs_v.append(imgs_comb_v)
		imgs_comb_v = Image.fromarray( imgs_comb_v)
			#imgs_comb_v.save(str(v) + 'Trifecta_vertical.jpg' )


	imgs_comb_final = np.vstack( (np.asarray(i) for i in imgs_v ) )
	imgs_comb_final = Image.fromarray( imgs_comb_final)#.convert('BGR') 
	#imgs_comb_final.save('Trifecta_final.jpg' )
	imgs_comb_final = np.array(imgs_comb_final)[: ,: , ::-1].copy()
	#cv2.imwrite('a.png',imgs_comb_final)
	out.write(imgs_comb_final)
	#clip = ImageClip(imgs_comb_final)
	
	
out.release()
