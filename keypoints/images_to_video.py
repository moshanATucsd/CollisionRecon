import numpy as np
from PIL import Image

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


for time in range(1000):

	list_im = []
	for num,value in enumerate(index):
		filename = folder_path + str(value) + '/labelled_image/' + str(-diff[num]).zfill(5) + '.png'
		if value == 6 or value == 10 or value == 16:
			continue
		list_im.append(filename)


	imgs    = [Image.open(i) for i in list_im ]
	min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]

	imgs_v = []
	for v in range(4):
		imgs_new = imgs[v*4:v*4 + 4]
		imgs_comb_v = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs_new ) )
		imgs_v.append(imgs_comb_v)
		imgs_comb_v = Image.fromarray( imgs_comb_v)
		imgs_comb_v.save(str(v) + 'Trifecta_vertical.jpg' )


	imgs_comb_final = np.vstack( (np.asarray(i) for i in imgs_v ) )
	imgs_comb_final = Image.fromarray( imgs_comb_final)
	imgs_comb_final.save('Trifecta_final.jpg' )
	
