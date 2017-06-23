import logging
#from multiprocessing import Pool
import time
import sys,os
import numpy as np
import gzip
import pickle
import networkx as nx
sys.path.append(os.getcwd() + "/libs/OpenSfM/")
from libs.OpenSfM.opensfm import exif

from libs.OpenSfM.opensfm import io
from libs.OpenSfM.opensfm import features
from libs.OpenSfM.opensfm import config
from libs.OpenSfM.opensfm import matching
from libs.OpenSfM.opensfm import reconstruction as recon
import dataset

import cv2
import glob

def save_features(filepath,config, image, points, descriptors, colors=None):
    io.mkdir_p(filepath)
    feature_type = config.get('feature_type')
    if ((feature_type == 'AKAZE' and config.get('akaze_descriptor') in ['MLDB_UPRIGHT', 'MLDB']) or
        (feature_type == 'HAHOG' and config.get('hahog_normalize_to_uchar', False))):
        feature_data_type = np.uint8
    else:
        feature_data_type = np.float32
    filepath = filepath + '/' + image
    np.savez_compressed(filepath,
             points=points.astype(np.float32),
             descriptors=descriptors.astype(feature_data_type),
             colors=colors)
def plot_matches(im1, im2, p1, p2):
    h1, w1, c = im1.shape
    h2, w2, c = im2.shape
    image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=im1.dtype)
    image[0:h1, 0:w1, :] = im1
    image[0:h2, w1:(w1 + w2), :] = im2

    p1 = features.denormalized_image_coordinates(p1, w1, h1)
    p2 = features.denormalized_image_coordinates(p2, w2, h2)
    pl.imshow(image)
    for a, b in zip(p1, p2):
        pl.plot([a[0], b[0] + w1], [a[1], b[1]], 'c')

    pl.plot(p1[:, 0], p1[:, 1], 'ob')
    pl.plot(p2[:, 0] + w1, p2[:, 1], 'ob')


def feature_extract(images,base_dir):
    ## Features extraction
    for camera,files in enumerate(images):
        for num,filenames in enumerate(files):
            p_unsorted, f_unsorted, c_unsorted = features.extract_features(io.imread(filenames),config)
            size = p_unsorted[:, 2]
            order = np.argsort(size)
            p_sorted = p_unsorted[order, :]
            f_sorted = f_unsorted[order, :]
            c_sorted = c_unsorted[order, :]
            save_features(base_dir+ '/features/',config,str(camera) + '_' + filenames.split('/')[-1],p_sorted,f_sorted,c_sorted)
            if config.get('matcher_type', 'FLANN') == 'FLANN':
                index = features.build_flann_index(f_sorted, config)
                index.save(base_dir+ '/features/'+str(camera) + '_' + filenames.split('/')[-1]+'.flann')
def load_features(config, featurename):
    feature_type = config.get('feature_type')
    s = np.load(featurename)
    if feature_type == 'HAHOG' and config.get('hahog_normalize_to_uchar', False):
         descriptors = s['descriptors'].astype(np.float32)
    else:remaining_images
         descriptors = s['descriptors']
    return s['points'], descriptors, s['colors'].astype(float)

def feature_matching(images,base_dir):
    for j in range(len(images)):
        filenames = images[j][50]
        featurename1 = base_dir+ '/features/' + str(j) + '_' + filenames.split('/')[-1] 
        
        p1, f1, c1= load_features(config,featurename1+'.npz')
        index1 = cv2.flann_Index()# if context.OPENCV3 else cv2.flann_Index()
        index1.load(f1,featurename1+'.flann')
        im1_matches = {}

        for match in range(len(images)):
            if match == j:
                continue
            image_name = str(match) + '_' + filenames.split('/')[-1]
            featurename2 = base_dir+ '/features/' + str(match) + '_' + filenames.split('/')[-1] 
            p2, f2, c2= load_features(config,featurename2+'.npz')
            index2 = cv2.flann_Index()# if context.OPENCV3 else cv2.flann_Index()
            index2.load(f2,featurename2+'.flann')
            matches = matching.match_symmetric(f1, index1, f2, index2, config)
            rmatches = matching.robust_match_fundamental(p1, p2, matches,config)
            im1_matches[image_name] = rmatches

        matches_path = base_dir+ '/matches/'
        io.mkdir_p(matches_path)
        with gzip.open(os.path.join(matches_path, '{}_matches.pkl.gz'.format(str(j) + '_' + filenames.split('/')[-1] )), 'wb') as fout:
            pickle.dump(im1_matches, fout)   

def save_tracks_graph(fileobj, graph):
    for node, data in graph.nodes(data=True):
        if data['bipartite'] == 0:
            image = node
            for track, data in graph[image].items():
                x, y = data['feature']
                fid = data['feature_id']
                r, g, b = data['feature_color']
                fileobj.write('%s\t%s\t%d\t%g\t%g\t%g\t%g\t%g\n' % (
                    str(image), str(track), fid, x, y, r, g, b))  

def load_tracks_graph(fileobj):
    g = nx.Graph()
    for line in fileobj:
        image, track, observation, x, y, R, G, B = line.split('\t')
        g.add_node(image, bipartite=0)
        g.add_node(track, bipartite=1)
        g.add_edge(
            image, track,
            feature=(float(x), float(y)),
            feature_id=int(observation),
            feature_color=(float(R), float(G), float(B)))
    return g


def incremental_reconstruction(config):
    """Run the entire incremental reconstruction pipeline."""
    graph_path = os.path.join(base_dir, 'tracks.csv')
    with open(graph_path) as fin:
        graph = load_tracks_graph(fin)
    tracks, images = matching.tracks_and_images(graph)
    print(tracks)
    remaining_images = set(images)
    gcp = None
    common_tracks = matching.all_common_tracks(graph, tracks)
    reconstructions = []
    pairs = recon.compute_image_pairs(common_tracks, config)
    for im1, im2 in pairs:
        if im1 in remaining_images and im2 in remaining_images:
            tracks, p1, p2 = im1_matchesrecon.common_tracks[im1, im2]
            reconstruction = recon.bootstrap_reconstruction(data, graph, im1, im2, p1, p2)
            if reconstruction:
                remaining_images.remove(im1)
                remaining_images.remove(im2)
                reconstruction = grow_reconstruction(
                    data, graph, reconstruction, remaining_images, gcp)
                reconstructions.append(reconstruction)
                reconstructions = sorted(reconstructions,
                                         key=lambda x: -len(x.shots))
                data.save_reconstruction(reconstructions)

def feature_tracks(images,base_dir):
    features = {}
    colors = {}
    for j in range(len(images)):
        filenames = images[j][50]
        featurename1 = base_dir+ '/features/' + str(j) + '_' + filenames.split('/')[-1] 
        
        p, f, c= load_features(config,featurename1+'.npz')
        features[j] = p[:, :2]
        colors[j] = c
        
    matches = {}
    for i in range(len(images)):
        filenames = images[i][50]
        matches_path = base_dir+ '/matches/'
        im1 = str(i) + '_' + filenames.split('/')[-1]
        matches_file = os.path.join(matches_path, '{}_matches.pkl.gz'.format(im1))
        try:
            with gzip.open(matches_file, 'rb') as fin:
                im1_matches = pickle.load(fin)
        except IOError:
               continue
        for im2 in im1_matches:
               matches[i, im2] = im1_matches[im2]

    tracks_graph = matching.create_tracks_graph(features, colors, matches,config)       
    os.path.join(base_dir, 'tracks.csv')
    with open(os.path.join(base_dir, 'tracks.csv'), 'w') as fout:
            save_tracks_graph(fout, tracks_graph)   
def exif_data(images,base_dir):
    for camera,files in enumerate(images):
        for num,filenames in enumerate(files):
            d = exif.extract_exif_from_file(filenames)
        # Image Height and Image Width
            if d['width'] <= 0 or not data.config['use_exif_size']:
                d['height'], d['width'] = io.imread(filenames).shape[:2]
            ext = os.path.join(base_dir, 'exif')
            io.mkdir_p(ext)
            with open(os.path.join(ext, '{}.exif'.format(str(camera) + '_' + filenames.split('/')[-1] )), 'w') as fout:
                io.json_dump(d, fout)
                
if __name__ == "__main__":
    base_dir = '/home/dinesh/CarCrash/data/CarCrash/Test'
    data = dataset.DataSet(base_dir)
    config = data.config
    inputs = range(3)
    images = []
    masks = []
    for i,num in enumerate(inputs):
        image_list = sorted(glob.glob(base_dir + '/'+ str(i) + '/images/*.png'))
        mask_list = sorted(glob.glob(base_dir + '/'+ str(i) + '/mask/*'))
        if len(mask_list) != len(image_list):
            print('mask Not available')
        images.append(image_list)
        masks.append(mask_list)
    

    exif_data(images,base_dir)       
    #feature_extract(images,base_dir)
    #feature_matching(images,base_dir)
    
    feature_tracks(images,base_dir)
    graph_path = os.path.join(base_dir, 'tracks.csv')
    with open(graph_path) as fin:
        graph = load_tracks_graph(fin)
    
    tracks, images = matching.tracks_and_images(graph)
    remaining_images = set(images)
    gcp = None
    common_tracks = matching.all_common_tracks(graph, tracks)
    reconstructions = []
    pairs = recon.compute_image_pairs(common_tracks, config)
    for im1, im2 in pairs:
        if im1 in remaining_images and im2 in remaining_images:
            tracks, p1, p2 = common_tracks[im1, im2]
            reconstruction = recon.bootstrap_reconstruction(data, graph, im1, im2, p1, p2)
            if reconstruction:
                remaining_images.remove(im1)
                remaining_images.remove(im2)
                reconstruction = grow_reconstruction(
                    data, graph, reconstruction, remaining_images, gcp)
                reconstructions.append(reconstruction)
                reconstructions = sorted(reconstructions,
                                         key=lambda x: -len(x.shots))
                data.save_reconstruction(reconstructions)
    #incremental_reconstruction(config)
    #num_cores = multiprocessing.cpu_count()-2
    #results = Parallel(n_jobs=num_cores)(delayed(Read)(i,videofile,folder_path,output_path,time) for i in inputs)
