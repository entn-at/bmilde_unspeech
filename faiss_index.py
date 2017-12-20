#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:10:24 2017

@author: milde
"""

import faiss
import numpy as np
import argparse
import kaldi_io


def get_utt_info(myid, utt_ids, utt_map, pos_map):
    return utt_ids[utt_map[myid]], pos_map[myid]

def index_train(feat_filename, index_filename, index_type="IVF1024,Flat"):
    
    print('Loading feature data...')
    max_feats = 10000
    feats, utt_ids = kaldi_io.readArk(feat_filename , limit=max_feats)
    
    complete_feat_len = 0
    for feat in feats:
        complete_feat_len += feat.shape[0]
    
    utt_map = np.zeros(complete_feat_len, dtype=np.int32)
    pos_map = np.zeros(complete_feat_len, dtype=np.int32)
    
    # create the utt_map and pos_map that map a pos id to utt id and position inside an utterance
    pos=0
    for i,(feat, utt_id) in enumerate(zip(feats, utt_ids)):
        for j in range(feat.shape[0]):
            utt_map[pos] = i 
            pos_map[pos] = j
            pos += 1
            
    complete_feats = np.concatenate(feats, axis=0)
    
    print('Going to index features of shape:',complete_feats.shape)
    
    index = faiss.index_factory(complete_feats[0].shape[1], index_type)
    
    index.train(complete_feats)
    
    print('Indexing finished.')
    
    search_vec_index =  100
    search_vec = complete_feats[search_vec_index]
    
    D, I =  index.search(search_vec, 10)
    
    print("Neighboors of ", search_vec_index)
    
    print("D:")
    
    print(D)
    
    print("I:")
    
    print(I)

def index_load(featfile, index_filename):
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a faiss search index that can be used to do efficient similarity search.')
    parser.add_argument('-i', '--input_featfile', dest='featfile', help='The feature file to visualize.', type=str, default = 
                        '/Users/milde/inspect/feats_transVgg16big_win50_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size1024_unit_norm_var_dropout_keep0.9_batchnorm_bndecay0.999_l2_reg0.0005_dot_combine/dev/feats.ark')
    parser.add_argument('-f', '--format', dest='format', help='Format of the feature file (raw,kaldi_ark)', type=str, default = 'kaldi_ark')
    parser.add_argument('-x', '--index', dest='index', help='Filename of the index', type=str, default = 'models/index1')
    parser.add_argument('-t', '--index_type', dest='index_type', help='Index factory string that defines the tzpe of index. See github.com/facebookresearch/faiss for details.', type=str, default = 'IVF1024,Flat')
    #parser.add_argument('-w', '--window_size', dest='window_size', help='Window size', type=int, default = 32)
    #parser.add_argument('-r', '--frame_rate', dest='frame_rate', help='Frames per second', type=int, default = 100)

#    parser.add_argument('-m', '--max_frames', dest='max_frames', help='Maximum frames', type=int, default = 10000)
#    parser.add_argument('-p', '--phn_file', dest='phn_file', help='Phoneme annotation file', type=str, default = '')
    parser.add_argument('--mode', dest='mode', help='(load|train)', type=str, default = 'train')

    args = parser.parse_args()
    
    if args.mode=='train':
        index_train(args.featfile, args.index, args.window_size, args.indextype)
    elif args.mode=='featshow':
        index_load(args.featfile, args.index)
    else:
        print("mode not supported.")
