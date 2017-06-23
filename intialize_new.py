import logging
#from multiprocessing import Pool
import time
import sys,os
import numpy as np
import gzip
import pickle
import networkx as nx
import time
sys.path.append(os.getcwd() + "/libs/OpenSfM/")
from libs.OpenSfM.opensfm import exif

from libs.OpenSfM.opensfm import io
from libs.OpenSfM.opensfm import features
from libs.OpenSfM.opensfm import config
from libs.OpenSfM.opensfm import matching
from libs.OpenSfM.opensfm import reconstruction as recon
import dataset
from multiprocessing import Pool
import logging

logger = logging.getLogger(__name__)

import cv2
import glob

def detect(args):
    image, data = args
    logger.info('Extracting {} features for image {}'.format(
        data.feature_type().upper(), image))

    if not data.feature_index_exists(image):
        mask = data.mask_as_array(image)
        if mask is not None:
            logger.info('Found mask to apply for image {}'.format(image))
        preemptive_max = data.config.get('preemptive_max', 200)
        p_unsorted, f_unsorted, c_unsorted = features.extract_features(
            data.image_as_array(image), data.config, mask)
        if len(p_unsorted) == 0:
            return

        size = p_unsorted[:, 2]
        order = np.argsort(size)
        p_sorted = p_unsorted[order, :]
        f_sorted = f_unsorted[order, :]
        c_sorted = c_unsorted[order, :]
        p_pre = p_sorted[-preemptive_max:]
        f_pre = f_sorted[-preemptive_max:]
        data.save_features(image, p_sorted, f_sorted, c_sorted)
        data.save_preemptive_features(image, p_pre, f_pre)

        if data.config.get('matcher_type', 'FLANN') == 'FLANN':
            index = features.build_flann_index(f_sorted, data.config)
            data.save_feature_index(image, index)
def detect_feature(data):
    images = data.images()
    arguments = [(image, data) for image in images]
    start = time.time()
    processes = data.config.get('processes', 1)
    if processes == 1:
        for arg in arguments:
            detect(arg)
            print('1')
    else:
        p = Pool(processes)
        p.map(detect, arguments)
    end = time.time()
    with open(data.profile_log(), 'a') as fout:
        fout.write('detect_features: {0}\n'.format(end - start))

if __name__ == "__main__":
    base_dir = '/home/dinesh/CarCrash/data/CarCrash/Test/'
    data = dataset.DataSet(base_dir)
    #detect_feature(data)
            
