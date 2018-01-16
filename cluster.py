#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:36:53 2017

@author: Benjamin Milde
"""

from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing

import numpy as np
import utils

def feats2sumlen(feats):
    sumlen = 0
    for uttid, feat in zip(feats):
        sumlen += len(feat)
    return sumlen

def cluster_phn(n_clusters, wav_files, ark_file, hopping_size, window_size, subsample, n_jobs=10):
    
    feats = utils.readArk(ark_file)
    
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


def cluster_speaker(ark_file, dbscan_eps=0.5, dbscan_min_samples=2, utt_2_spk = None):
    
    uttids, feats = utils.readArk(ark_file)
    
    feats = np.vstack(feats)
    
    print('feats shape:', feats.shape)
    print()
    
    ground_truth_utt_2_spk, ground_truth_utt_2_spk_int = None,None
    
    if utt_2_spk is not None:
        ground_truth_utt_2_spk = [utt_2_spk[utt_id] for utt_id in uttids]
        
        le = preprocessing.LabelEncoder()
        le.fit(ground_truth_utt_2_spk)
        
        ground_truth_utt_2_spk_int = le.transform(ground_truth_utt_2_spk)
        
        print("Ground truth speaker classes available:")
    
    dbscan_algo = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, n_jobs=10)
    
    clustering = dbscan_algo.fit_predict(feats)
    
    print('Numpy bincount:', np.bincount(clustering))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--n-clusters', dest='n_clusters', help='The number of clusters.', type=int, default = 42)
    parser.add_argument('-f', '--wav-files', dest='wav_files', help='Original wav files. Kaldi format file, uttid -> wavfile.', type=str, default = '')
    parser.add_argument('-a', '--input-ark', dest='ark_file', help='Input ark with the computed features.', type=str, default = '')
    parser.add_argument('-h', '--hopping-size', dest='hopping_size', help='Hopping size, in ms.', type=int, default=-1)
    parser.add_argument('-w', '--window-size', dest='window_size', help='Window size, in ms.', type=int, default=-1)
    parser.add_argument('-s', '--subsampling', dest='subsample', help='Subsample the input corpus (probability to take input frame from 0.0 to 1.0). Set to 1.0 to take all the data', type=float, default=0.01)
    parser.add_argument('-r', '--sampling-rate', dest='samplingrate', help='Sampling rate of the corpus', type=int, default=16000)


    args = parser.parse_args()
    
    cluster(**args)