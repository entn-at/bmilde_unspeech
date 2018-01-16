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

def visualize_stats(feat_filename, max_feats, abs_feats=True, reverse_sort=True):
    feats, utt_ids = kaldi_io.readArk(feat_filename , limit=max_feats)
    
    feats_len=len(feats)
    
    print("Loaded:" + str(feats_len) + "feats")
    
    sums=[]
    for feat in feats:
        if abs_feats:
            feat = np.abs(feat)
        local_sum=np.sum(feat, axis=0) / float(len(feat))
        print(local_sum.shape)
        sums.append(local_sum)
    
    sums=np.stack(sums,axis=0)
    print(sums.shape)
    
    finalsum=np.sum(sums, axis=0) / float(feats_len)
    
    finalsum_sorted=np.sort(np.array(finalsum))
    
    if reverse_sort:
        finalsum_sorted = finalsum_sorted[::-1]
    
    print(finalsum)
    print(finalsum_sorted)
    
    
    plt.plot(finalsum_sorted)
    
    plt.figure(1)
    plt.matshow([finalsum])
    plt.figure(2)
    plt.matshow([finalsum_sorted])
    plt.show()
    
def visualize_kaldi_bin_feats(feat_filename, max_frames, num_feat=8, phn_file='', phn_offset=5, wav_file='', do_tsne=False):
    feats, utt_ids = kaldi_io.readArk(feat_filename , limit=10)
    
    print(feats, utt_ids)
    
    print('showing features for utt_id:', utt_ids[num_feat])

    print('min vector:')
    print(np.min(feats[num_feat], axis=1))
    print('max vector:')
    print(np.max(feats[num_feat], axis=1))
    print('sum vector:')
    print(np.sum(feats[num_feat], axis=1))

    print(feats[num_feat].shape)

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

    
    if do_tsne:
        plt.figure(1)
        
        print('Calculating TSNE:')
        model = TSNE(n_components=2, random_state=0)
    
        tsne_data = model.fit_transform(feats[num_feat])
        plt.plot(tsne_data[:,0], tsne_data[:,1], '--')
    
        print('Now showing tsne plot:')
        plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature file viewer, for different formats. (raw, kaldi text, unsupervised challenge 2017 format).')
    parser.add_argument('-i', '--input_featfile', dest='featfile', help='The feature file to visualize.', type=str, default = 
                        #'/Users/milde/inspect/feats_transVgg16big_win50_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_dropout_keep0.9_l2_reg0.0005_dot_combine/dev/feats.ark')
                        #'/Users/milde/inspect/feats_transVgg16big_win50_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size1024_dropout_keep0.9_batchnorm_bndecay0.95_l2_reg0.0005_dot_combine/dev/feats.ark')      
                        #'/Users/milde/inspect/kaldi_train/feats.normalized.ark')
                        #'/Users/milde/inspect/feats_transVgg16big_win50_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size1024_unit_norm_var_dropout_keep0.9_batchnorm_bndecay0.999_l2_reg0.0005_dot_combine/dev/feats.ark')
                        'feats_vgg.ark')
    parser.add_argument('-f', '--format', dest='format', help='Format of the feature file (raw,kaldi_ark)', type=str, default = 'kaldi_ark')
    parser.add_argument('-m', '--max_frames', dest='max_frames', help='Maximum frames', type=int, default = 200)
    parser.add_argument('-n', '--num_feat', dest='num_feat', help='feat file to visualize', type=int, default = 0)
    parser.add_argument('-p', '--phn_file', dest='phn_file', help='Phoneme annotation file', type=str, default = '')
    parser.add_argument('--mode', dest='mode', help='(featshow|stats)', type=str, default = 'featshow')


    args = parser.parse_args()
    
    if args.mode=='stats':
        visualize_stats(args.featfile, args.max_frames)
    elif args.mode=='featshow':
        visualize_kaldi_bin_feats(args.featfile, args.max_frames, phn_file= args.phn_file, num_feat=args.num_feat)
    else:
        print("mode not supported.")
        #visualize_stats(args.featfile, args.max_frames)
    
    #if args.format == 'kaldi_ark':
    #    
