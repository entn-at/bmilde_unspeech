#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:26:30 2017

@author: me
"""

import pylab as  plt
import numpy as np
import argparse
import kaldi_io
import utils

from sklearn.manifold import TSNE

framerate = 100.0
samplerate = 16000.0
samples_per_frame = samplerate / framerate

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
    
def visualize_kaldi_bin_feats(feat_filename, max_frames, num_feat=0, phn_file='', phn_offset=5, wav_file=''):
    feats, utt_ids = kaldi_io.readArk(feat_filename , limit=10)
    
    print(feats, utt_ids)
    
    print('showing features for utt_id:', utt_ids[num_feat])



    if phn_file == '':
        plt.matshow(feats[num_feat][:max_frames].T)  
        plt.show()
    else:
        plt.matshow(feats[num_feat][:max_frames].T)  
        positions,names = utils.loadPhnFile(phn_file)
        xpositions = [float(pos[1])/samples_per_frame - phn_offset for pos in positions if float(pos[1])/samples_per_frame < max_frames]
        for xc in xpositions:
            plt.axvline(x=xc, color='k', linestyle='--')
        plt.show()

    print('Calculating TSNE:')
    model = TSNE(n_components=2, random_state=0)

    print('Now showing tsne plot:')
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature file viewer, for different formats. (raw, kaldi text, unsupervised challenge 2017 format).')
    parser.add_argument('-i', '--input_featfile', dest='featfile', help='The feature file to visualize.', type=str, default = '')
    parser.add_argument('-f', '--format', dest='format', help='Format of the feature file (raw,kaldi_ark)', type=str, default = 'kaldi_ark')
    parser.add_argument('-m', '--max_frames', dest='max_frames', help='Maximum frames', type=int, default = 100)
    parser.add_argument('-p', '--phn_file', dest='phn_file', help='Phoneme annotation file', type=str, default = '')

    args = parser.parse_args()
    
    if args.format == 'kaldi_ark':
        visualize_kaldi_bin_feats(args.featfile, args.max_frames, phn_file= args.phn_file)
