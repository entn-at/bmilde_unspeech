#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:26:30 2017

@author: me
"""

import pylab as  plt
import numpy as np
import argparse

feat = "1009.fea"
max_frames = 100

def visualize_text_feats(feat_filename, max_frames, lineoffset=0):
    i = 0
    feats = []
    with open(feat_filename) as feat_in:
        for line in feat_in:
            i += 1
            if line[-1]=='\n':
                line = line[:-1]
            frame = np.asarray([float(elem) for elem in line.split(' ')])[lineoffset:]
            print(frame)
            if i == max_frames:
                break
            feats.append(frame)
            
    plt.matshow(feats)
    plt.show()
    
def visualize_kaldi_bin_feats(feat_filename, num_feat=0):
    feats, utt_ids = kaldi_io.readArk(feat_filename , limit=10)
    
    print('showing features for utt_id:', utt_ids[num_feat])

    plt.matshow(feats[num_feat][max_frames])  
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature file viewer, for different formats. (raw, kaldi text, unsupervised challenge 2017 format).')
    parser.add_argument('-i', '--input_featfile', dest='featfile', help='The feature file to visualize.', type=str, default = '')
    parser.add_argument('-f', '--format', dest='format', help='Format of the feature file (raw,kaldi_ark)', type=str, default = 'kaldi_ark')

    args = parser.parse_args()
    
    if args.format = 'kaldi_ark':
        visualize_kaldi_bin_feats(args.featfile)
