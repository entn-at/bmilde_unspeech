#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:36:53 2017

@author: Benjamin Milde
"""

from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE

from sklearn import preprocessing
from sklearn import metrics

import pylab as plt
import numpy as np
import utils
import kaldi_io
import random

from numpy.core.umath_tests import inner1d

from scipy.spatial import distance

import argparse

def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))

def feats2sumlen(feats):
    sumlen = 0
    for uttid, feat in zip(feats):
        sumlen += len(feat)
    return sumlen

def cluster_phn(n_clusters, wav_files, ark_file, hopping_size, window_size, subsample, n_jobs=4):
    
    feats = kaldi_io.readArk(ark_file)
    
    # preallocate the array and establish array sizes
    inner_dim = feats[0][0].shape[1]
    sum_len = feats2sumlen(feats)
    
    feats_flat = np.zeros(sum_len, inner_dim)
    uttids_flat = []
    pos_flat = []
    
    pos = 0
    for uttid, feat in zip(feats):
        feats_flat[pos:pos+feat.shape[0]] = feat
        pos += feat.shape[0]
        #repeating uttid feat.shape[0] times
        uttids_flat += [uttid]*feat.shape[0]
        pos_flat = np.array(float(x) for x in range(feat.shape[0])) * hopping_size
   
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_jobs=n_jobs).fit(feats_flat)
    kmeans.labels_

# assumes that 
def pos_neg_dot_distance(a, b):
    half_index=int(a.shape[0] / 2)
    return sigmoid(np.dot(a[:half_index],b[half_index:]))

# assumes that 
def pairwise_pos_neg_dot_distance(a, b):
    half_index=int(a.shape[0] / 2)
    return sigmoid(np.dot(a[:half_index],b[half_index:])) + sigmoid(np.dot(b[:half_index],a[half_index:]))

def cluster_speaker(ark_file, dbscan_eps=0.0005, dbscan_min_samples=3, utt_2_spk = None, tsne_viz=True, n_jobs=4, range_search=False):
    
    feats, uttids = kaldi_io.readArk(ark_file)
    
    print('feat[0] shape: ', feats[0].shape)
    
    feats = np.vstack([feat[0] for feat in feats])
    
    print('feats shape:', feats.shape)
    print('feat[0] shape: ', feats[0].shape)

    print('halfindex:', feats[0].shape[0] / 2)
    
    print('some distances:')
    for a,b in [(random.randint(0, len(feats)), random.randint(0, len(feats))) for i in range(10)] + [(0,0)]:
        dst = distance.euclidean(feats[a],feats[b])
        print('euc dst:', a,b,'=',dst)
        dst = distance.cosine(feats[a],feats[b])
        print('cos dst:', a,b,'=',dst)
        dst = np.dot(feats[a],feats[b])
        print('dot dst:', a,b,'=',dst)
        
        dst = pos_neg_dot_distance(feats[a],feats[b])
        print('pos_neg_dot_distance dst:', a,b,'=',dst)
        
        
        dst = pairwise_pos_neg_dot_distance(feats[a],feats[b])
        print('pairwise_pos_neg_dot_distance dst:', a,b,'=',dst)
        
    
        
    for a in range(50):
        for b in range(50):
            print('feats[a]:',feats[a])
            print('feats[b]:',feats[b])
            dst = pos_neg_dot_distance(feats[a],feats[b])
            print('pos_neg_dot_distance dst:', a,b,'=',dst)
            pairwise_pos_neg_dot_distance(feats[a],feats[b])
            print('pairwise_pos_neg_dot_distance dst:', a,b,'=',dst)
    
    ground_truth_utt_2_spk, ground_truth_utt_2_spk_int = None,None
    
    if utt_2_spk is not None:
        ground_truth_utt_2_spk = [utt_2_spk[utt_id] for utt_id in uttids]
        
        le = preprocessing.LabelEncoder()
        le.fit(ground_truth_utt_2_spk)
        
        ground_truth_utt_2_spk_int = le.transform(ground_truth_utt_2_spk)
        
        print("Ground truth speaker classes available:")
        
        print(ground_truth_utt_2_spk_int)
    
    print('Now running DBSCAN clustering.')
    
    if range_search:
        for dbscan_eps in [0.00005, 0.0005, 0.005, 0.05, 0.5, 5, 50, 500]:
            for dbscan_min_samples in [1, 2, 3, 4, 5, 6, 7, 8, 10, 100]:
                
                dbscan_algo = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric=pairwise_pos_neg_dot_distance, n_jobs=1)
                clustering = dbscan_algo.fit(feats)
                clustering_labels = list(clustering.labels_)
                
                print('dbscan_eps', dbscan_eps, 'dbscan_min_samples', dbscan_min_samples)
                print('num clusters:', len(set(clustering_labels)))
    
    dbscan_algo = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric=pairwise_pos_neg_dot_distance, n_jobs=1)
    clustering = dbscan_algo.fit(feats)
    clustering_labels = list(clustering.labels_)
                
    print('dbscan_eps', dbscan_eps, 'dbscan_min_samples', dbscan_min_samples)
    print('num clusters:', len(set(clustering_labels)))
    print(clustering_labels)
    
    #print('Numpy bincount of the clustering:', np.bincount(clustering))
    
    if tsne_viz:
        print('Calculating TSNE:')
        model = TSNE(n_components=2, random_state=0, metric='cosine')#pos_neg_dot_distance)

        tsne_data = model.fit_transform(feats) #[feat[:100] for feat in feats])

        num_speakers = max(clustering_labels)+2
        
        colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
        colorst = colormap(np.linspace(0, 0.9, num_speakers)) #[colormap(i) for i in np.linspace(0, 0.9, num_speakers)]  
        
        cs = [colorst[clustering_labels[i]] for i in range(len(clustering_labels))]
        
        print(tsne_data[:,0])
        print(tsne_data[:,1])
        
        plt.scatter(tsne_data[:,0], tsne_data[:,1], color=cs)

        print('Now showing tsne plot:')
        plt.show()
    
    if utt_2_spk is not None:
        ARI = metrics.adjusted_rand_score(ground_truth_utt_2_spk_int, clustering)
        print('ARI score:', ARI)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--n-clusters', dest='n_clusters', help='The number of clusters, if kmeans is used.', type=int, default = 42)
    parser.add_argument('-f', '--wav-files', dest='wav_files', help='Original wav files. Kaldi format file, uttid -> wavfile.', type=str, default = '')
    parser.add_argument('-a', '--input-ark', dest='ark_file', help='Input ark with the computed features.', type=str, default = 'feats/feats_transResnet_v2_101_nsampling_same_spk_win64_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0005_dot_combine_tied_embs/dev/feats.ark')
    parser.add_argument('-hs', '--hopping-size', dest='hopping_size', help='Hopping size, in ms.', type=int, default=-1)
    parser.add_argument('-w', '--window-size', dest='window_size', help='Window size, in ms.', type=int, default=-1)
    parser.add_argument('-s', '--subsampling', dest='subsample', help='Subsample the input corpus (probability to take input frame from 0.0 to 1.0). Set to 1.0 to take all the data', type=float, default=0.01)
    parser.add_argument('-r', '--sampling-rate', dest='samplingrate', help='Sampling rate of the corpus', type=int, default=16000)
    parser.add_argument('--utt2spk', dest='utt2spk', help='Needed to compare speaker clusters and calculate scores.', type=str, default = None)
    parser.add_argument('--mode', dest='mode', help='(cluster_speaker|cluster_phn)', type=str, default = 'cluster_speaker')

    args = parser.parse_args()
    
    if args.mode == 'cluster_phn':
        cluster_phn(n_clusters=args.n_clusters, wav_files = args.wav_files, ark_file=args.ark_file, 
                    hopping_size=args.hopping_size, window_size=args.window_size, subsample=args.subsample, n_jobs=4)
    elif args.mode == 'cluster_speaker':
        cluster_speaker(args.ark_file, dbscan_eps=0.5, dbscan_min_samples=5, utt_2_spk = args.utt2spk)
    else:
        print('mode:', args.mode, 'not supported.')