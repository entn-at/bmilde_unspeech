#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:36:53 2017

@author: Benjamin Milde
"""

from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE

from hdbscan import HDBSCAN

from sklearn import preprocessing
from sklearn import metrics

import pylab as plt
import numpy as np
import utils
import kaldi_io
import random

import tensorflow as tf

#from numpy.core.umath_tests import inner1d

from scipy.spatial import distance

import argparse

def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))

def feats2sumlen(feats):
    sumlen = 0
    for uttid, feat in zip(feats):
        sumlen += len(feat)
    return sumlen


def load_feats_flat(ark_file):
    
    feats, uttids = kaldi_io.readArk(ark_file)
    
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
        
    return feats, feats_flat, uttids_flat, pos_flat


def cluster_rnn_phn(n_clusters, wav_files, ark_file, hopping_size, window_size, subsample, n_jobs=4):
    
    feats, uttids = kaldi_io.readArk(ark_file)
    
    tf.segment_mean()
    #tf.
    
    #from https://github.com/tensorflow/tensorflow/issues/7389
    ones = tf.ones_like(x)
    count = tf.unsorted_segment_sum(ones, ids, 2)
    sums = tf.unsorted_segment_sum(x, ids, 2)
    mean = tf.divide(sums, count)
    

    
    


def cluster_phn(n_clusters, wav_files, ark_file, hopping_size, window_size, subsample, n_jobs=4):
    
    feats, feats_flat, uttids_flat, pos_flat = load_feats_flat(ark_file)
   
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_jobs=n_jobs).fit(feats_flat)
    kmeans.labels_

# assumes that 
def pos_neg_dot_distance(a, b):
    half_index=int(a.shape[0] / 2)
    return 1.0 - sigmoid(np.dot(a[:half_index],b[half_index:]))

# assumes that 
def pairwise_pos_neg_dot_distance(a, b):
    half_index=int(a.shape[0] / 2)
    return 1.0 - 0.5*(sigmoid(np.dot(a[:half_index],b[half_index:])) + sigmoid(np.dot(b[:half_index],a[half_index:])))

def pairwise_normalize(a):
    half_index=int(a.shape[0] / 2)
    norm1 = np.linalg.norm(a[:half_index], ord=2)
    norm2 = np.linalg.norm(a[half_index:], ord=2)
    
    return np.hstack([a[:half_index]/norm1, a[half_index:]/norm2])

def cluster_speaker(ark_file, dbscan_eps=0.0005, dbscan_min_samples=3, utt_2_spk = None, output_utt_2_spk = None, tsne_viz=False, n_jobs=4, range_search=False):
    
    print('Loading feats now:')

    feats, uttids = kaldi_io.readArk(ark_file)
    
    print('feat[0] shape: ', feats[0].shape)
    
    #feats = np.vstack([pairwise_normalize(feat[0]) for feat in feats])
    
    print('Generating mean vector.')

    feats = np.vstack([feat.mean(0) for utt,feat in zip(uttids,feats)])

    print('Done. feats shape:', feats.shape)

#    feats = np.vstack([feat[0] for utt,feat in zip(uttids,feats) if 'AlGore' not in utt])
#    uttids = [utt for utt in uttids if 'AlGore' not in utt]
    
    print('feats shape:', feats.shape)
    print('feat[0] shape: ', feats[0].shape)

    print('halfindex:', feats[0].shape[0] / 2)
    
    print('some distances:')
    for a,b in [(random.randint(0, len(feats)-1), random.randint(0, len(feats)-1)) for i in range(10)] + [(0,0)]:
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
        
    
        
    for a in range(10):
        for b in range(10):
            print('feats[a]:',feats[a])
            print('feats[b]:',feats[b])
            dst = pos_neg_dot_distance(feats[a],feats[b])
            print('pos_neg_dot_distance dst:', a,b,'=',dst)
            pairwise_pos_neg_dot_distance(feats[a],feats[b])
            print('pairwise_pos_neg_dot_distance dst:', a,b,'=',dst)
    
    ground_truth_utt_2_spk, ground_truth_utt_2_spk_int = None,None
    
    if utt_2_spk is not None:
        utt_2_spk = utils.loadUtt2Spk(utt_2_spk)
        
        ground_truth_utt_2_spk = [utt_2_spk[utt_id] for utt_id in uttids]
        
        le = preprocessing.LabelEncoder()
        le.fit(ground_truth_utt_2_spk)
        
        ground_truth_utt_2_spk_int = le.transform(ground_truth_utt_2_spk)
        
        print("Ground truth speaker classes available:")
        
        print(ground_truth_utt_2_spk_int)
    
    print('Now running DBSCAN clustering on', len(uttids),'entries.')
    
    bestARI = 0.0
    bestConf ={}
    
    if range_search:
        
        eps_range = [x/100.0 for x in range(1,100)]
        min_samples_range = [1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 50, 100]
        
        result_mat = np.zeros((len(eps_range), len(min_samples_range)))
        
        print('shape result mat:', result_mat.shape)
        
        for i_eps,dbscan_eps in enumerate(eps_range):
            for i_min_samples, dbscan_min_samples in enumerate(min_samples_range):
                
                dbscan_algo = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric=pairwise_pos_neg_dot_distance, n_jobs=1)
                clustering = dbscan_algo.fit(feats)
                clustering_labels = list(clustering.labels_)
                
                print('dbscan_eps', dbscan_eps, 'dbscan_min_samples', dbscan_min_samples)
                print('num clusters:', len(set(clustering_labels)))
                
                ARI = metrics.adjusted_rand_score(ground_truth_utt_2_spk_int, clustering_labels)
                
                result_mat[i_eps][i_min_samples] = float(ARI)
                
                print('ARI:', ARI)
                
                if ARI > bestARI:
                    print('Found new best conf:', ARI)
                    bestConf = {'eps': dbscan_eps,'min_samples': dbscan_min_samples }
                    bestARI = ARI
                    
        plt.matshow(result_mat)
        plt.show()
        
        np.save(ark_file + '.dbrangescan_cluster_ARI',  result_mat)
                    
    print('bestARI:', bestARI)
    print('bestConf:',bestConf)
    
    #cluster_algo = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric=pairwise_pos_neg_dot_distance, n_jobs=28)
    cluster_algo = HDBSCAN(min_cluster_size=10, metric='euclidean', algorithm='best', core_dist_n_jobs=28)
    clustering = cluster_algo.fit(feats)
    clustering_labels = list(clustering.labels_)
                
    print('dbscan_eps', dbscan_eps, 'dbscan_min_samples', dbscan_min_samples)
    print('num clusters:', len(set(clustering_labels)))
    print(clustering_labels)
    
    #print('Numpy bincount of the clustering:', np.bincount(clustering))
    
    if tsne_viz:
        print('Calculating TSNE:')
        
        model = TSNE(n_components=2, random_state=0, metric=pos_neg_dot_distance)
        tsne_data = model.fit_transform(feats)
        
        #model = TSNE(n_components=2, random_state=0, metric='cosine')
        #tsne_data = model.fit_transform([feat[100:] for feat in feats])

        if utt_2_spk is not None:
            num_speakers = max(ground_truth_utt_2_spk_int) +1
        else:
            num_speakers = len(set(clustering_labels))
        
        colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
        colorst = colormap(np.linspace(0, 0.9, num_speakers)) #[colormap(i) for i in np.linspace(0, 0.9, num_speakers)]  
        
        if utt_2_spk is not None:
            cs = [colorst[ground_truth_utt_2_spk_int[i]] for i in range(len(clustering_labels))]
        else:
            cs = [colorst[clustering_labels[i]] for i in range(len(clustering_labels))]
        
        print(tsne_data[:,0])
        print(tsne_data[:,1])
        
        plt.scatter(tsne_data[:,0], tsne_data[:,1], color=cs)
        #for i,elem in enumerate(tsne_data):
        #    print(cs[0])
        #    print(ground_truth_utt_2_spk[0])
        #    plt.scatter(elem[0], elem[1], color=cs[i], label=ground_truth_utt_2_spk[i])
        plt.legend()
        
  #      for i in range(tsne_data.shape[0]):
  #          plt.text(tsne_data[i,0], tsne_data[i,1], uttids[i], fontsize=8, color=cs[i])

        print('Now showing tsne plot:')
        plt.show()
    
    if utt_2_spk is not None:
        ARI = metrics.adjusted_rand_score(ground_truth_utt_2_spk_int, clustering_labels)
        print('ARI score:', ARI)
        
    if output_utt_2_spk is not None:
        print('Saving result to:', output_utt_2_spk)
        with open(output_utt_2_spk, 'w') as output_utt_2_spk_out:
            for utt, label in zip(uttids,clustering_labels):
                output_utt_2_spk_out.write(utt + (' spk_outlier' if label == -1 else ' spk%05d' % label) + '\n' )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--n-clusters', dest='n_clusters', help='The number of clusters, if kmeans is used.', type=int, default = 42)
    parser.add_argument('-f', '--wav-files', dest='wav_files', help='Original wav files. Kaldi format file, uttid -> wavfile.', type=str, default = '')
    parser.add_argument('-a', '--input-ark', dest='ark_file', help='Input ark with the computed features.', type=str, default = 'feats/feats_transVgg16big_nsampling_same_spk_win64_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0005_featinput_unnormalized.feats.ark_dot_combine_tied_embs/train/feats.ark')
    parser.add_argument('-hs', '--hopping-size', dest='hopping_size', help='Hopping size, in ms.', type=int, default=-1)
    parser.add_argument('-w', '--window-size', dest='window_size', help='Window size, in ms.', type=int, default=-1)
    parser.add_argument('-s', '--subsampling', dest='subsample', help='Subsample the input corpus (probability to take input frame from 0.0 to 1.0). Set to 1.0 to take all the data', type=float, default=0.01)
    parser.add_argument('-r', '--sampling-rate', dest='samplingrate', help='Sampling rate of the corpus', type=int, default=16000)
    parser.add_argument('--utt2spk', dest='utt2spk', help='Needed to compare speaker clusters and calculate scores.', type=str, default = 'feats/tedlium/train/utt2spk')
    parser.add_argument('--output_utt2spk', dest='output_utt2spk', help='Where to store speaker output speaker clusters.', type=str, default = 'feats/tedlium/train/cluster_utt2spk')
    parser.add_argument('--mode', dest='mode', help='(cluster_speaker|cluster_phn)', type=str, default = 'cluster_speaker')

    args = parser.parse_args()
    
    if args.mode == 'cluster_phn':
        cluster_phn(n_clusters=args.n_clusters, wav_files = args.wav_files, ark_file=args.ark_file, 
                    hopping_size=args.hopping_size, window_size=args.window_size, subsample=args.subsample, n_jobs=4)
    elif args.mode == 'cluster_speaker':
        cluster_speaker(args.ark_file, dbscan_eps=0.05, dbscan_min_samples=5, utt_2_spk = args.utt2spk, output_utt_2_spk = args.output_utt2spk)
    else:
        print('mode:', args.mode, 'not supported.')
