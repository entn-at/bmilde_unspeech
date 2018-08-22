#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:36:53 2017

@author: Benjamin Milde
"""

license = '''

Copyright 2017,2018 Benjamin Milde (Language Technology, Universität Hamburg, Germany)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE

from hdbscan import HDBSCAN

import sklearn
from sklearn import preprocessing
from sklearn import metrics

from scipy.spatial.distance import pdist

import pylab as plt
#import numpy
import numpy as np
import utils
import kaldi_io
import random
import sys

import multiprocessing

from functools import partial

try:
    import faiss
    from gpu_dbscan.gpu_dbscan import GpuDbscan
    GpuDbscan_available = True
except:
    print("Warning could not import faiss, cannot use GPU clustering.")

import tensorflow as tf

#from numpy.core.umath_tests import inner1d

from scipy.spatial import distance

from collections import Counter

import argparse

#for cmudict
cmudict_silence  = ['SIL', 'NSN']
cmudict_vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
cmudict_consonants = ['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']

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
    
results_table_header = ["feat_key","cl_size", "min_s", "clusters_train", "clusters_test", "clusters_dev", "outliers_train", "outliers_test", "outliers_dev", "ARI_train", "ARI_test",
                             "ARI_dev", "NMI_train", "NMI_test", "NMI_dev", "prec_train", "prec_dev", "prec_test", "recall_train",
                             "recall_dev", "recall_test", "f1_train", "f1_dev", "f1_test"]

results_table_header_dict = dict([(x[1],x[0]) for x in enumerate(results_table_header)])

def save_result(feat_key, result_key, result):
    
    results_file = 'cluster_results.csv'
    
    results_table_dict = {}
    try:
        with open('cluster_results.csv') as cluster_results:
            i = 0
            for line in cluster_results:
                if line[-1] == '\n':
                    line = line[:-1]
                split = line.split(',')
                if i!= 0:
                    results_table_dict[split[0]] = split[1:]
                i+=1
    except:
        print(results_file, 'does not exist, creating a new one.')
                    
    if feat_key in results_table_dict:
        results_table_dict[feat_key][results_table_header_dict[result_key]] = result
    else:
        results_table_dict[feat_key] = ['-'] * len(results_table_header)
        results_table_dict[feat_key][results_table_header_dict[result_key]] = result

    with open('cluster_results.csv', 'w') as cluster_results:        
        cluster_results.write(','.join(results_table_header) + '\n')
        for key in sorted(results_table_dict.keys()):
            cluster_results.write(','.join([key]+results_table_dict[key]) + '\n')

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

def cluster_dbscan_gpu(X, eps, min_samples):
    d=X.shape[-1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    print("Building faiss index, d=",d)
    index = faiss.GpuIndexFlatIP(res, d, flat_config)
    index.add(X)
    print("Running dbscan.")
    gd = GpuDbscan(X, gpu_index=index)
    cluster_ids, core_point_flag, visited_flag = gd.gpu_dbscan(1, 10)

    return cluster_ids

def same_different_experiment(ark_file, utt_2_spk, half_index=-1, normalize=False, fileset='', use_metric='cosine', max_spks=-1, random_seed=42):
    
    results_file = 'samedifferent_results.csv'
    
    from pyannote.metrics.plot.binary_classification import plot_det_curve, plot_distributions
    
    print('Loading feats now:')

    feats, uttids = kaldi_io.readArk(ark_file.replace('%set', fileset))
    
    print('loaded: ' + str(len(feats)) + ' feats')
    print('feat[0] shape: ', feats[0].shape)
    
    #feats = np.vstack([pairwise_normalize(feat[0]) for feat in feats])
    
    print('Generating mean vector.')

    feats = np.vstack([feat.mean(0) for utt,feat in zip(uttids,feats)])
    
    if half_index != -1:
        print('Cutting vectors at ', half_index, 'and normalize to unit length' if normalize else '')
        feats = np.vstack([feat[:half_index]/(np.linalg.norm(feat[:half_index]) if normalize else 1.0) for feat in feats])
    else:
        if normalize:
            print('Normalize to unit length.')
            feats = np.vstack([feat/np.linalg.norm(feat) for feat in feats])
    
    #print(type(feats))
    #print(feats)

    if utt_2_spk is not None and utt_2_spk.lower() != 'none' and utt_2_spk.strip() != '':
        utt_2_spk = utils.loadUtt2Spk(utt_2_spk.replace('%set', fileset))
        
        if max_spks != -1:
            
            # sample subset of speakers
            spk_set = list(set(utt_2_spk.values()))
       
            # make speaker selection reproducable
            spk_set.sort()
            #print('Selecting from these speakers:', spk_set)
            random.seed(random_seed)
            #np.random.seed(42)
            
            selected_spks = random.sample(spk_set, min(len(spk_set),max_spks))
            
            print('Selecting random subset of speakers with random seed',random_seed,':', len(selected_spks),'speakers')
            print(selected_spks)        
    
            # make new utt2spk dictionary on subset
            utt_2_spk_new = dict([(key,utt_2_spk[key]) for key in utt_2_spk if utt_2_spk[key] in selected_spks])
            
            #filter feats and uttids
            feats = [feat for feat, uttid in zip(feats, uttids) if uttid in utt_2_spk_new]
            uttids = [uttid for uttid in uttids if uttid in utt_2_spk_new]
            
            print('Reduced feats to: ' + str(len(feats)) + ' feats')
            print('Reduced uttids to: ' + str(len(feats)) + ' uttids')
           
            utt_2_spk = utt_2_spk_new
        else:
            print('Using all speakers:',len(set(utt_2_spk.values())))
        
        ground_truth_utt_2_spk = [utt_2_spk[utt_id] for utt_id in uttids]
        
        le = preprocessing.LabelEncoder()
        le.fit(ground_truth_utt_2_spk)
        
        ground_truth_utt_2_spk_int = le.transform(ground_truth_utt_2_spk)
        
        print("Ground truth speaker classes available:")
        
        print(ground_truth_utt_2_spk_int)
        
    print('Calculating',use_metric,'distance matrix...') 
    #print('feats shape:', feats.shape)       
    distances = pdist(feats, metric=use_metric)

    print('Calculating ground thruth distance matrix...')
    y_true = pdist(np.asarray(ground_truth_utt_2_spk_int)[:,np.newaxis], metric='chebyshev') < 1
    
    result_key = ark_file.split('/')[-3] +  ('.' + fileset if fileset != '' else '')  + '.' + use_metric
    prefix = 'plots/plot.' + ark_file.split('/')[-3] + '.' + use_metric + ('.' + fileset if fileset != '' else '') + '.seed_' + str(random_seed)
    
    plot_distributions(y_true, distances, prefix, xlim=(0, 2), ymax=3, nbins=100)
    
    eer = plot_det_curve(y_true, -distances, prefix)
    
    print('EER = {eer:.2f}%'.format(eer=100*eer))
    
    with open(results_file,'a') as outfile:
        outfile.write(result_key + ' ' + '{eer:.2f}%'.format(eer=100*eer) + '\n')

def load_phones_txt(phones_txt, remove_begin_end_markers=True, ignore_diamond=True, ignore_eps=True, ignore_nsn=True):
    print('Loading: ', phones_txt, 'now.')
    phones = {}
    with open(phones_txt) as phones_file:
        for line in phones_file:
            split = line.split()
            phone = split[0]
            if remove_begin_end_markers:
                if '_' in phone:
                    phone = phone.split('_')[0]
            if ignore_diamond:
                if '#' in phone:
                    continue
                
            if ignore_eps:
                if 'eps' in phone:
                    continue
                
 #           if ignore_nsn:
 #               if 'NSN' in phone:
 #                   continue
                
            phone_num = int(split[1])
            phones[phone_num] = phone
    return phones

def load_all_ctm_pos(ctm_file, num2phoneid=None, limit=-1, epsilon=0.001):
    all_ctm = {}
    with open(ctm_file) as ctm:
        last_utt_id = ''
        utt_phones = []
        utt_pos = []
        utt_local_len = 0
        utt_no = 0
        for line in ctm:
            
            if limit != -1 and utt_no > limit:
                break
            
            split = line.split()
            myid, pos_start, length, phone = split[0], int(float(split[2]) * 100.0 + epsilon), int(float(split[3]) * 100.0 + epsilon), int(split[4])
            
            # When the utterance id changes, we create a flat version of the phones, so that the position in the numpy array == frame position
            if myid != last_utt_id and len(utt_phones) > 0:
                all_ctm[last_utt_id] = (np.array(utt_pos), np.array(utt_phones))
                print('loaded:', last_utt_id) #, utt_phones_flat_stacked)
                utt_phones = []
                utt_pos = []
                utt_local_len = 0
                utt_no += 1
                
            #print(line[:-1])
            #print(myid, pos_start, length, num2phone[phone])
            
            #print(pos_start, utt_local_len)
            assert(pos_start == utt_local_len)
            
            if num2phoneid is None:
                utt_phones += [phone]
            else:
                utt_phones += [num2phoneid[phone]]
                
            utt_pos += [pos_start]
            
            last_utt_id = myid
            utt_local_len += length
        
    return all_ctm

def load_all_ctm_flat(ctm_file, num2phoneid=None, limit=-1, epsilon=0.001):
    all_ctm_flat = {}
    with open(ctm_file) as ctm:
        last_utt_id = ''
        utt_phones_flat = []
        utt_local_len = 0
        utt_no = 0
        for line in ctm:
            
            if limit != -1 and utt_no > limit:
                break
            
            split = line.split()
            myid, pos_start, length, phone = split[0], int(float(split[2]) * 100.0 + epsilon), int(float(split[3]) * 100.0 + epsilon), int(split[4])
            
            # When the utterance id changes, we create a flat version of the phones, so that the position in the numpy array == frame position
            if myid != last_utt_id and len(utt_phones_flat) > 0:
                utt_phones_flat_stacked = np.hstack(utt_phones_flat)
                all_ctm_flat[last_utt_id] = utt_phones_flat_stacked
                print('loaded:', last_utt_id) #, utt_phones_flat_stacked)
                utt_phones_flat = []
                utt_local_len = 0
                utt_no += 1
                
            #print(line[:-1])
            #print(myid, pos_start, length, num2phone[phone])
            
            #print(pos_start, utt_local_len)
            assert(pos_start == utt_local_len)
            
            if num2phoneid is None:
                utt_phones_flat += [np.array([phone] * length, dtype=np.int16)]
            else:
                utt_phones_flat += [np.array([num2phoneid[phone]] * length, dtype=np.int16)]
                
            last_utt_id = myid
            utt_local_len += length
        
    return all_ctm_flat


def get_phone_mutual_info(phone, feats, uttids, all_ctm_db, do_mutual_info_classif=True):
    feat_vars = []
    phone_vars = []
    for feat, uttid in zip(feats, uttids):
        pos_array, phone_array = all_ctm_db[uttid]
        
        feat_vars += [feat[list(pos_array)]]
        
        if phone != -1:
            phone_var = (phone_array == phone) * 1.0
        else:
            phone_var = phone
            
        phone_vars += [phone_var]
    
    feat_vars = np.vstack(feat_vars)
    phone_vars = np.hstack(phone_vars)
    
    print(feat_vars.shape)
    print(phone_vars.shape)
    
    if do_mutual_info_classif:
        return sklearn.feature_selection.mutual_info_classif( feat_vars , phone_vars )
    else:
        return (feat_vars , phone_vars)

def bigrams(sequence):
    return zip(sequence, sequence[1:])

def trigrams(sequence):
    return zip(sequence, sequence[1:], sequence[2:])

def fourgrams(sequence):
    return zip(sequence, sequence[1:], sequence[2:], sequence[3:])

def fivegrams(sequence):
    return zip(sequence, sequence[1:], sequence[2:], sequence[3:], sequence[4:])

def sixgrams(sequence):
    return zip(sequence, sequence[1:], sequence[2:], sequence[3:], sequence[4:],  sequence[5:])

def tsne(ark_file, alignment_dir, fileset='train', limit=4000, subsample=1000, only_common_bigrams=True, cmudict_sort=True, normalize=True):

    feats, uttids = kaldi_io.readArk(ark_file.replace('%set', fileset), limit=limit)
    
    if normalize:
        print('Normalize to unit length.')
        feats = [feat/np.linalg.norm(feat) for feat in feats]
    
    phones_txt =  alignment_dir + 'phones.txt'
    num2phone_orig = load_phones_txt(phones_txt)
    #phone2num_orig = 
    
    print(num2phone_orig)
    
    phoneset = list(set(num2phone_orig.values()))
    
    if cmudict_sort:
        num2phone = dict(enumerate(cmudict_silence + cmudict_vowels + cmudict_consonants))
        phone2num = dict([(y,x) for x,y in enumerate(cmudict_silence + cmudict_vowels + cmudict_consonants)])
    else:
        num2phone = dict(enumerate(sorted(list(set(num2phone_orig.values())))))
        phone2num = dict([(y,x) for x,y in enumerate(sorted(list(set(num2phone_orig.values()))))])

    num2phoneid = dict([(i,phone2num[x]) for i,x in num2phone_orig.items()])
    print(num2phoneid)
    
    ctm_file = alignment_dir + 'all.ctm'
    
    all_ctm_db = load_all_ctm_pos(ctm_file, num2phoneid, limit=limit)
    
    phn_ngram_counts = Counter()
    
    ngram_func = fivegrams 
    
    feat_vars = []
    phone_vars = []
    for feat, uttid in zip(feats, uttids):
        pos_array, phone_array = all_ctm_db[uttid]
        
        assert(len(pos_array)==len(phone_array))
        
        feat_vars += [feat[list(pos_array)]]
        phone_vars += [phone_array]
        
        phn_ngram_counts.update(ngram_func(phone_array))
    
    feat_vars = np.vstack(feat_vars)
    phone_vars = np.hstack(phone_vars)
    
    #print(phn_counts)
    
    most_common_ngrams = []
    
    print('Most common phone ngrams:')
    for phone_ngram, count in phn_ngram_counts.most_common(10):
        print(phone_ngram, [num2phone[elem] for elem in phone_ngram], count)
        most_common_ngrams.append(phone_ngram)
    
    #most_common_ngrams = [(30, 12, 30, 4, 26), (32, 4, 27, 35, 11), (4, 37, 9, 26, 19)] 
    #print('Manually selected:', most_common_ngrams)
    
    most_common_bigrams_to_num = dict([(b,a) for a,b in enumerate(most_common_ngrams)])
    
    print('Loaded', len(phone_vars), 'phoneme examples')
    
    if only_common_bigrams:
        select_pos = [i for i,elem in enumerate(ngram_func(phone_vars)) if elem in most_common_ngrams]    
        
        feat_vars = feat_vars[select_pos]
        phone_vars = np.asarray([most_common_bigrams_to_num[elem] for i,elem in enumerate(ngram_func(phone_vars)) if elem in most_common_ngrams]) #phone_vars[select_pos]
    
        print(phone_vars)
        print(feat_vars)
    
        assert(len(feat_vars)==len(phone_vars))
    
    if subsample != -1:
        feat_phone_vars_sampled_idx = np.random.choice(np.arange(len(feat_vars)), subsample, replace=False)
        feat_vars = feat_vars[feat_phone_vars_sampled_idx]
        phone_vars = phone_vars[feat_phone_vars_sampled_idx]
        
        print('Subsampled to:', len(phone_vars), 'phoneme examples')
    
    model = TSNE(n_components=2, random_state=0, metric='cosine')
    tsne_data = model.fit_transform(feat_vars)
    
    num_classes = len(phoneset)
    
    if only_common_bigrams:
        num_classes = len(list(set(phone_vars)))
    
    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
    colorst = colormap(np.linspace(0, 1.0, num_classes+1)) #[colormap(i) for i in np.linspace(0, 0.9, num_speakers)]  
       
    cs = [colorst[phone_vars[i]] for i in range(len(phone_vars))]
    
    #print(tsne_data[:,0])
    #print(tsne_data[:,1])
    
    for i,color in enumerate(colorst):
        plt.text(30+float(i)*3,30, str(i), color=color)
    
    plt.scatter(tsne_data[:,0], tsne_data[:,1], color=cs)
    #for i,elem in enumerate(tsne_data):
    #    print(cs[0])
    #    print(ground_truth_utt_2_spk[0])
    #    plt.scatter(elem[0], elem[1], color=cs[i], label=ground_truth_utt_2_spk[i])
    #plt.legend(most_common_ngrams)
    
    plt.show()

def corr_phn(ark_file, alignment_dir, fileset='train', limit=5000, cmudict_sort=True):
  
        
    print('len cmudict arpabet:', len(cmudict_silence + cmudict_vowels + cmudict_consonants))
    
    feats, uttids = kaldi_io.readArk(ark_file.replace('%set', fileset), limit=limit)
    
    phones_txt =  alignment_dir + 'phones.txt'
    num2phone_orig = load_phones_txt(phones_txt)
    #phone2num_orig = 
    
    print(num2phone_orig)
    
    phoneset = list(set(num2phone_orig.values()))
    
    not_vowels = []
    for elem in phoneset:
        if elem not in cmudict_vowels:
            not_vowels += [elem]
            
    for elem in cmudict_vowels:
        if elem not in phoneset:
            print('vowel NA:', elem)
    
    print(not_vowels)
    
    print('Loaded num2phone_orig for: ',len(set(num2phone_orig.values())),'phones')

    if cmudict_sort:
        num2phone = dict(enumerate(cmudict_silence + cmudict_vowels + cmudict_consonants))
        phone2num = dict([(y,x) for x,y in enumerate(cmudict_silence + cmudict_vowels + cmudict_consonants)])
    else:
        num2phone = dict(enumerate(sorted(list(set(num2phone_orig.values())))))
        phone2num = dict([(y,x) for x,y in enumerate(sorted(list(set(num2phone_orig.values()))))])
    
    num2phoneid = dict([(i,phone2num[x]) for i,x in num2phone_orig.items()])
    print(num2phoneid)
    
    ctm_file = alignment_dir + 'all.ctm'
    
    all_ctm_db = load_all_ctm_pos(ctm_file, num2phoneid, limit=limit)
    
    phone_len = len(num2phone.keys())
    
    feat_dim = len(feats[0][0])
    
    print(feat_dim)
    
    phone_mutual_info_classifs = []
    
    do_2d = True
    parallel = True
    
    if do_2d:
        if not parallel:
            for phone in sorted(num2phone.keys()):
                
                #for feat_num in range(feat_dim):
                phone_mutual_info_classif = get_phone_mutual_info(phone, feats, uttids, all_ctm_db)
                phone_mutual_info_classifs += [phone_mutual_info_classif]
            
            phone_mutual_info_classifs = np.vstack(phone_mutual_info_classifs)      
        else:
            pool = multiprocessing.Pool(28)
            get_phone_mutual_info_partial = partial(get_phone_mutual_info, feats=feats, uttids=uttids, all_ctm_db=all_ctm_db)
            phone_mutual_info_classifs = pool.map(get_phone_mutual_info_partial, sorted(num2phone.keys()))
            
        #for feat_num in range(feat_dim):
    else:
        feat_vars = []
        phone_vars = []
        for feat, uttid in zip(feats, uttids):
            ctm = all_ctm_db[uttid]
            
            #print('lens (ctm|feat):',len(ctm), len(feat))
            if len(ctm) > len(feat):
                print(uttid, 'warning, ctm is slightly longer than feature len:', len(ctm), 'vs.', len(feat))
                ctm = ctm[:len(feat)]
            elif len(ctm) < len(feat):
                print(uttid, 'warning, ctm is slightly shorter than feature len:', len(ctm), 'vs.', len(feat))
            
            assert(len(feat) == len(ctm))
            
            feat_vars += [feat]
            
            #phone_var = #(np.array(ctm) == phone) * 1.0
            phone_vars += [ctm]
        
        feat_vars = np.vstack(feat_vars)
        phone_vars = np.hstack(phone_vars)
        
        print(feat_vars.shape)
        print(phone_vars.shape)
        
        phone_mutual_info_classifs = sklearn.feature_selection.mutual_info_classif( feat_vars , phone_vars )
        
        print(phone_mutual_info_classifs)
    
    print(plt.rcParams["figure.figsize"])
    fig = plt.figure(figsize = (11,7))
    ax = fig.add_subplot(111)
    
    phone_mutual_info_classifs[0] *= 0.7
    
    ax.matshow(phone_mutual_info_classifs, aspect='auto')
    
    plt.yticks(sorted(num2phone.keys()), [num2phone[x] for x in sorted(num2phone.keys())])
    plt.show()
        

#def cluster_speaker(ark_file, dbscan_eps=0.0005, dbscan_min_samples=3, utt_2_spk = None, output_utt_2_spk = None, tsne_viz=False, n_jobs=4, range_search=False, use_gpu=True):
def cluster_speaker(ark_file, half_index=-1, dbscan_eps=0.0005, dbscan_min_samples=3,     min_cluster_sizes_str = "5",
    min_samples_str = "3", utt_2_spk = None, output_utt_2_spk = None, fileset= 'dev', tsne_viz=False, n_jobs=4,
    db_scan_range_search=False, hdb_scan_range_search=False, normalize=True, do_save_result=True, use_gpu=False):

    postfix=''
    print('Loading feats now:')

    feats, uttids = kaldi_io.readArk(ark_file.replace('%set', fileset))
    
    print('feat[0] shape: ', feats[0].shape)
    
    #feats = np.vstack([pairwise_normalize(feat[0]) for feat in feats])
    
    print('Generating mean vector.')

    feats = np.vstack([feat.mean(0) for utt,feat in zip(uttids,feats)])

    if half_index != -1:
        print('Cutting vectors at ', half_index, 'and normalize to unit length' if normalize else '')
        feats = np.vstack([feat[half_index:]/(np.linalg.norm(feat[half_index:]) if normalize else 1.0) for feat in feats])
    else:
        if normalize:
            print('Normalize to unit length.')
            feats = np.vstack([feat/np.linalg.norm(feat) for feat in feats])

    print('Done. feats shape:', feats.shape)

#    feats = np.vstack([feat[0] for utt,feat in zip(uttids,feats) if 'AlGore' not in utt])
#    uttids = [utt for utt in uttids if 'AlGore' not in utt]
    
    print('feats shape:', feats.shape)
    print('feat[0] shape: ', feats[0].shape)

    print('halfindex:', half_index)
    
#    print('some distances:')
#    for a,b in [(random.randint(0, len(feats)-1), random.randint(0, len(feats)-1)) for i in range(10)] + [(0,0)]:
#        dst = distance.euclidean(feats[a],feats[b])
#        print('euc dst:', a,b,'=',dst)
#        dst = distance.cosine(feats[a],feats[b])
#        print('cos dst:', a,b,'=',dst)
#        dst = np.dot(feats[a],feats[b])
#        print('dot dst:', a,b,'=',dst)
#        
#        dst = pos_neg_dot_distance(feats[a],feats[b])
#        print('pos_neg_dot_distance dst:', a,b,'=',dst)
#        
#        
#        dst = pairwise_pos_neg_dot_distance(feats[a],feats[b])
#        print('pairwise_pos_neg_dot_distance dst:', a,b,'=',dst)    
#        
#    for a in range(10):
#        for b in range(10):
#            print('feats[a]:',feats[a])
#            print('feats[b]:',feats[b])
#            dst = pos_neg_dot_distance(feats[a],feats[b])
#            print('pos_neg_dot_distance dst:', a,b,'=',dst)
#            pairwise_pos_neg_dot_distance(feats[a],feats[b])
#            print('pairwise_pos_neg_dot_distance dst:', a,b,'=',dst)
    
    ground_truth_utt_2_spk, ground_truth_utt_2_spk_int = None,None
    
    if utt_2_spk is not None and utt_2_spk.lower() != 'none' and utt_2_spk.strip() != '':
        utt_2_spk = utils.loadUtt2Spk(utt_2_spk.replace('%set', fileset))
        
        ground_truth_utt_2_spk = [utt_2_spk[utt_id] for utt_id in uttids]
        
        le = preprocessing.LabelEncoder()
        le.fit(ground_truth_utt_2_spk)
        
        ground_truth_utt_2_spk_int = le.transform(ground_truth_utt_2_spk)
        
        print("Ground truth speaker classes available:")
        
        print(ground_truth_utt_2_spk_int)
    
    print('Now running DBSCAN clustering on', len(uttids),'entries.')
    
    bestARI = 0.0
    bestConf ={}
    
    if db_scan_range_search:
        
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
        
        np.save(ark_file + '.dbrangescan_cluster_ARI' + postfix,  result_mat)
                    
        print('bestARI:', bestARI)
        print('bestConf:',bestConf)
    
#<<<<<<< HEAD
#    #min_cluster_sizes = [2,3,4,5,6,7,8,9,10,11,12]
#    #min_samples = [2,3,4,5,6,7,8,9,10]
#=======
#    if use_gpu:
#        clustering_labels = cluster_dbscan_gpu(feats, dbscan_eps, dbscan_min_samples)
#    else:
#        #cluster_algo = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric=pairwise_pos_neg_dot_distance, n_jobs=28)
#        cluster_algo = HDBSCAN(min_cluster_size=10, metric='euclidean', algorithm='best', core_dist_n_jobs=28)
#        clustering = cluster_algo.fit(feats)
#        clustering_labels = list(clustering.labels_)
#                
#    print('dbscan_eps', dbscan_eps, 'dbscan_min_samples', dbscan_min_samples)
#    print('num clusters:', len(set(clustering_labels)))
#    print(clustering_labels)
#>>>>>>> 3ae8c233f81b69d8a9ff183eaa939f8490155225
    
    min_cluster_sizes = [int(x) for x in min_cluster_sizes_str.split(',')]
    min_samples = [int(x) for x in min_samples_str.split(',')]
    
    result_mat = np.zeros((len(min_cluster_sizes), len(min_samples)))
    result_mat_outliers = np.zeros_like(result_mat)
    result_mat_n = np.zeros_like(result_mat)
    
    best_pairwise_f1 = 0.0
    bestConf ={}
    
    # previous good config: min_cluster_size=5, min_samples=3
    for i,min_cluster_size in enumerate(min_cluster_sizes):
        for j,min_sample in enumerate(min_samples):
            
            feat_key = ark_file.split('/')[-3] + '_' + str(min_cluster_size) + '_' + str(min_sample)
            
            if  do_save_result:
                save_result(feat_key, "cl_size", str(min_cluster_size))
                save_result(feat_key, "min_s", str(min_sample))
            
            print('Running HDBSCAN with min_cluster_size', min_cluster_size, 'min_samples', dbscan_min_samples)
            #cluster_algo = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric=pairwise_pos_neg_dot_distance, n_jobs=28)
            cluster_algo = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_sample, metric='euclidean', algorithm='best', core_dist_n_jobs=28)
            clustering = cluster_algo.fit(feats)
            clustering_labels = list(clustering.labels_)
                        
            print('Num clusters:', len(set(clustering_labels)))
            print(clustering_labels)
            
            sys.stdout.flush()
            
            #print('Numpy bincount of the clustering:', np.bincount(clustering))
        
            number_format = "%.4f"
    
            clustering_labels1 = clustering_labels
            clustering_labels2 = []
    
            num_outliers = -1
            for elem in clustering_labels1:
                if elem == -1:
                    clustering_labels2.append(num_outliers)
                    num_outliers -= 1
                else:
                    clustering_labels2.append(elem)
            
            num_outliers = (num_outliers+1)*-1
            
            if utt_2_spk is not None and utt_2_spk.lower() != 'none' and utt_2_spk.strip() != '':
                
                if do_save_result:
                    save_result(feat_key, 'outliers_'  + fileset, str(num_outliers))
                    save_result(feat_key, 'clusters_'  + fileset, str(len(set(clustering_labels))))
                    
                
                print('Number of outliers:',num_outliers, '(',number_format % (float(num_outliers)*100.0 / float(len(uttids))) ,'%)')
        
                ARI = metrics.adjusted_rand_score(ground_truth_utt_2_spk_int, clustering_labels)
                print('ARI score:', number_format % ARI)
                vmeasure = metrics.v_measure_score(ground_truth_utt_2_spk_int, clustering_labels)        
                print('V-measure:', number_format % vmeasure)
        
                ARI2 = metrics.adjusted_rand_score(ground_truth_utt_2_spk_int, clustering_labels2)
                print('ARI score (each outlier its own cluster):', number_format % ARI2)
                vmeasure2 = metrics.v_measure_score(ground_truth_utt_2_spk_int, clustering_labels2)
                print('V-measure (each outlier its own cluster):', number_format % vmeasure2)
        
                if do_save_result:
                    save_result(feat_key, 'ARI_'  + fileset, number_format % ARI2)
                    save_result(feat_key, 'NMI_'  + fileset, number_format % vmeasure2)
        
                print('Calculating pairwise recall:')
        
                cluster_pairwise = pdist(np.asarray(clustering_labels2)[:,np.newaxis], metric='chebyshev') < 1
                groundtruth_pairwise = pdist(np.asarray(ground_truth_utt_2_spk_int)[:,np.newaxis], metric='chebyshev') < 1
        
                #pairwise_recall = metrics.recall_score(groundtruth_pairwise, cluster_pairwise , pos_label=True, average='binary')
                #pairwise_precision = metrics.precision_score(groundtruth_pairwise, cluster_pairwise , pos_label=True, average='binary')
        
                #print('scikit learn recall / precision:', pairwise_recall, pairwise_precision)
        
                # efficient binary comparision, since the pairwise matrix can be huge for large n
                tp = np.sum(np.bitwise_and(groundtruth_pairwise, cluster_pairwise))
                fp = np.sum(np.bitwise_and(np.invert(groundtruth_pairwise), cluster_pairwise))
                fn = np.sum(np.bitwise_and(groundtruth_pairwise, np.invert(cluster_pairwise))) 
        
                pairwise_precision = tp / (tp+fp)
                pairwise_recall = tp / (tp+fn)
        
                pairwise_f1 = 2.0 * pairwise_recall * pairwise_precision / (pairwise_recall + pairwise_precision)
        
                print('pairwise recall / precision / f1-score (each outlier its own cluster):', number_format % pairwise_recall, number_format % pairwise_precision, number_format % pairwise_f1)
        
                if do_save_result:
                    save_result(feat_key, 'recall_'  + fileset, number_format % pairwise_recall)
                    save_result(feat_key, 'prec_'  + fileset, number_format % pairwise_precision)
                    save_result(feat_key, 'f1_'  + fileset, number_format % pairwise_f1)
        
                if pairwise_f1 > best_pairwise_f1:
                    print('Found new best pairwise f1:', pairwise_f1)
                    bestConf = {'min_cluster_size': min_cluster_size,'min_sample': min_sample, 'n':  len(set(clustering_labels)), 'outliers':num_outliers }
                    best_pairwise_f1 = pairwise_f1
        
                result_mat[i][j] = float(pairwise_f1)
                result_mat_outliers[i][j] = num_outliers
                result_mat_n[i][j] = len(set(clustering_labels))
        
                #print('pairwise recall / precision / f1-score:', number_format % pairwise_recall, number_format % pairwise_precision, number_format % pairwise_f1)
        
                print('Clustering predicted classes:' , len(set(clustering_labels)))
                print('Ground truth classes' , len(set(ground_truth_utt_2_spk_int)))

#    if len(min_cluster_sizes) > 1 or len(min_samples) > 1:
#        np.save(ark_file + '.hdbrangescan_cluster_f1' + postfix,  result_mat)
#
#        print('best f1:', best_pairwise_f1)
#        print(bestConf)
#
#        print('f1 scores:')
#        plt.matshow(result_mat)
#        plt.show()
#        
#        print('num outliers')
#        plt.matshow(result_mat_outliers)
#        plt.show()
#        
#        print('n')
#        plt.matshow(result_mat_n)
#        plt.show()
            
    if tsne_viz:
        print('Calculating TSNE:')
        
        model = TSNE(n_components=2, random_state=0, metric='euclidean')
        tsne_data = model.fit_transform(feats)
        
        #model = TSNE(n_components=2, random_state=0, metric='cosine')
        #tsne_data = model.fit_transform([feat[100:] for feat in feats])

        if utt_2_spk is not None and utt_2_spk.lower() != 'none' and utt_2_spk.strip() != '':
            num_speakers = max(ground_truth_utt_2_spk_int) +1
        else:
            num_speakers = len(set(clustering_labels))
        
        colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
        colorst = colormap(np.linspace(0, 0.9, num_speakers)) #[colormap(i) for i in np.linspace(0, 0.9, num_speakers)]  
        
        if utt_2_spk is not None and utt_2_spk.lower() != 'none' and utt_2_spk.strip() != '':
            cs = [colorst[ground_truth_utt_2_spk_int[i]] for i in range(len(clustering_labels))]
        else:
            cs = [colorst[clustering_labels[i]] for i in range(len(clustering_labels))]
        
        #print(tsne_data[:,0])
        #print(tsne_data[:,1])
        
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
    

    if output_utt_2_spk is not None:
        if len(min_cluster_sizes) > 1 or len(min_samples) > 1:
            print('Not saving clustering result, since we searched a full range. Rerun with a single min_cluster_size and min_samples parameter.')
        else:
            output_utt_2_spk = output_utt_2_spk.replace('%minclustersize', str(min_cluster_size))
            output_utt_2_spk = output_utt_2_spk.replace('%minsample', str(min_sample))
            output_utt_2_spk = output_utt_2_spk.replace('%set', fileset)
            featstr = ark_file.split('/')[-3]
            featstr = featstr.replace('featinput_unnormalized.feats.ark_dot_combine_tied_embs','std_end_conf').replace('feats_','')
            print('featstr:',featstr)
            output_utt_2_spk = output_utt_2_spk.replace('%feat',featstr)
            output_utt_2_spk += ('_l2norm' if normalize else '')
            #output_utt_2_spk += postfix
            print('Saving result to:', output_utt_2_spk)
            with open(output_utt_2_spk, 'w') as output_utt_2_spk_out:
                for utt, label in zip(uttids,clustering_labels2):
                    output_utt_2_spk_out.write(utt + (' spk%07d' % label).replace('-','o') + '\n' )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--n-clusters', dest='n_clusters', help='The number of clusters, if kmeans is used.', type=int, default = 42)
    parser.add_argument('-f', '--wav-files', dest='wav_files', help='Original wav files. Kaldi format file, uttid -> wavfile.', type=str, default = '')
    parser.add_argument('-a', '--input-ark', dest='ark_file', help='Input ark with the computed features.', type=str, default = 
                        #"feats/feats_transVgg16big_nsampling_rnd_win64_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size2048_unit_norm_var_dropout_keep0.9_l2_reg0.0005_featinput_unnormalized.feats.ark_dot_combine_tied_embs/%set/feats.ark")
                        #'feats/feats_transVgg16big_nsampling_rnd_win128_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size2048_unit_norm_var_dropout_keep0.9_l2_reg0.0005_featinput_unnormalized.feats.ark_dot_combine_tied_embs/%set/feats.ark')
                        #'feats/tedlium_ivectors_sp/%set/ivector_online.ark') #'feats/feats_transVgg16big_nsampling_same_spk_win64_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0005_featinput_unnormalized.feats.ark_dot_combine_tied_embs/test/feats.ark')
                        '/scratch/unspeech_diag/tedlium_train_diag_unspeech_phn_32_stride1/train_cleaned_sp_comb/ivector_online.ark')
                        #'/scratch/unspeech_diag/tedlium_train_diag_unspeech_no_2x_32_stride1/train_cleaned/ivector_online.ark')
    
    parser.add_argument('-hs', '--hopping-size', dest='hopping_size', help='Hopping size, in ms.', type=int, default=-1)
    parser.add_argument('-w', '--window-size', dest='window_size', help='Window size, in ms.', type=int, default=-1)
    parser.add_argument('-s', '--subsampling', dest='subsample', help='Subsample the input corpus (probability to take input frame from 0.0 to 1.0). Set to 1.0 to take all the data', type=float, default=0.01)
    parser.add_argument('-r', '--sampling-rate', dest='samplingrate', help='Sampling rate of the corpus', type=int, default=16000)
    parser.add_argument('--utt2spk', dest='utt2spk', help='Needed to compare speaker clusters and calculate scores.', type=str, default = 'feats/tedlium/%set/utt2spk_lium')
    parser.add_argument('--output_utt2spk', dest='output_utt2spk', help='Where to store speaker output speaker clusters.', type=str, default = 'feats/tedlium/%set/cl_utt2spk_min_cl%minclustersize_min_s_%minsample_%feat')
    parser.add_argument('--set', dest='set', help='e.g. (train|dev|test)', type=str, default = 'test')
    parser.add_argument('--mode', dest='mode', help='(cluster_speaker|cluster_phn|corr_phn|samedifferent)', type=str, default = 'tsne_phn')
    parser.add_argument('--hdbscan_min_cluster_sizes',  dest='hdbscan_min_cluster_sizes', help='hdbscan min_cluster_sizes parameter, either a single value or a comma separated list ov values.', default="5") #default="3,5,8")
    parser.add_argument('--hdbscan_min_samples_str', dest='hdbscan_min_samples_str', help='hdbscan min_samples_str parameter, either a single value or a comma separated list ov values.', default="3") #default="3,5,8")
    parser.add_argument('--half_index', dest='half_index', help='Cut the feature representation at a certain point (e.g. useful if you want to cut a combined pos/neg unspeech embedding vector), set to -1 to disable.',  type=int, default=-1)
    parser.add_argument('--max_spks', dest='max_spks', help='Maximum speakers', type=int, default=100)
    parser.add_argument('--metric', dest='metric', help='Metric to use for comparisions', type=str, default='euclidean')
    parser.add_argument('--random_seed',  dest='random_seed', help='Random seed for e.g. maximum speakers (so that the same speakers get selected accross experiments.)', type=int, default=42)
    parser.add_argument('--alignment_dir',dest='alignment_dir', help='Kaldi alignment directory, for correlating phones or for visualisation purposes. Should contain a all.ctm and phones.txt file. See https://www.eleanorchodroff.com/tutorial/kaldi/kaldi-forcedalignment.html',
                        type=str, default='/scratch/unspeech_diag/tri3_train_cleaned_diag/') 

    args = parser.parse_args()
    
    if args.mode == 'samedifferent':
        same_different_experiment(args.ark_file, utt_2_spk=args.utt2spk, half_index=args.half_index, fileset=args.set, max_spks=args.max_spks, use_metric=args.metric, random_seed=args.random_seed)
    elif args.mode == 'cluster_phn':
        cluster_phn(n_clusters=args.n_clusters, wav_files = args.wav_files, ark_file=args.ark_file, 
                    hopping_size=args.hopping_size, window_size=args.window_size, subsample=args.subsample, n_jobs=4)
    elif args.mode == 'corr_phn':
        corr_phn(ark_file=args.ark_file, alignment_dir=args.alignment_dir)
    elif args.mode == 'tsne_phn':    
        tsne(ark_file=args.ark_file, alignment_dir=args.alignment_dir)
    elif args.mode == 'cluster_speaker':
        cluster_speaker(args.ark_file, half_index=args.half_index , dbscan_eps=0.05, dbscan_min_samples=3, min_cluster_sizes_str = args.hdbscan_min_cluster_sizes,
    min_samples_str = args.hdbscan_min_samples_str, utt_2_spk = args.utt2spk, output_utt_2_spk = args.output_utt2spk, fileset = args.set)
    else:
        print('mode:', args.mode, 'not supported.')
