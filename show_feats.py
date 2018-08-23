#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:26:30 2017

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

import pylab as  plt
import numpy as np
import argparse
import kaldi_io
import utils
import random

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
    
def visualize_classes_tsne(feat_filename, utt_2_class_filename, half_index=-1, normalize=True, class_mean_vector=False):
    feats, utt_ids = kaldi_io.readArk(feat_filename, limit=25000)
    
    feats_len=len(feats)
    
    assert(len(utt_ids)==len(feats))
    
    print("Loaded:" + str(feats_len) + " feats.")
    
    feats = [feat.mean(0) for feat in feats]
    
    if half_index != -1:
        print('Cutting vectors at ', half_index, 'and normalize to unit length' if normalize else '')
        feats = [feat[:half_index]/(np.linalg.norm(feat[:half_index]) if normalize else 1.0) for feat in feats]
    else:
        if normalize:
            print('Normalize to unit length.')
            feats = [feat/np.linalg.norm(feat) for feat in feats]
            
    utt_2_class = utils.loadUtt2Spk(utt_2_class_filename)
    ground_truth_utt_2_class = [utt_2_class[utt_id] for utt_id in utt_ids if utt_id in utt_2_class]
    utt_ids_filtered = [utt_id for utt_id in utt_ids if utt_id in utt_2_class]
    #feats_filtered = [feat for feat,utt_id in zip(feats, utt_ids) if utt_id in utt_2_class]
    
    assert(len(ground_truth_utt_2_class) == len(utt_ids_filtered))
    #assert(len(utt_ids_filtered) == len(feats_filtered) )
    
    dataset = {}
    for feat,utt in zip(feats, utt_ids):
        if utt in utt_2_class:
            dataset[utt] = feat
    
    myclass_2_utt = {} 
    myclass_2_samples = {} 
    
    
    for myclass in set(ground_truth_utt_2_class):
        my_class_filtered_utts = [utt_id for utt_id, gd_class in zip(utt_ids_filtered, ground_truth_utt_2_class) if gd_class == myclass]
        if len(my_class_filtered_utts) > 100:
            myclass_2_utt[myclass] = my_class_filtered_utts
            myclass_2_samples[myclass] = random.sample(myclass_2_utt[myclass], min(1000,len(myclass_2_utt[myclass])))
        
    
    feats_samples = []
    feats_samples_classes = []
    
    if class_mean_vector:
        for myclass in myclass_2_samples:
            feats_samples += [np.vstack(dataset[utt] for utt in myclass_2_samples[myclass]).mean(0)]
            feats_samples_classes += [myclass]
    else:
        for myclass in myclass_2_samples:
            feats_samples += [dataset[utt] for utt in myclass_2_samples[myclass]]
            feats_samples_classes += [myclass]*len(myclass_2_samples[myclass])
            print('Added',len(myclass_2_samples[myclass]),'entries for',myclass)
            print([utt.replace('train-sample','train/sample') + '.mp3' for utt in myclass_2_samples[myclass]])
   
    class_2_num = dict([(a,b) for b,a in enumerate(list(myclass_2_samples.keys()))])
    
    print(class_2_num)
    
    feats_samples_classes_num = [class_2_num[myclass] for myclass in feats_samples_classes]
    
    #print(feats_samples_classes_num)
    
    num_classes = max(feats_samples_classes_num)
    print('Num classes=', num_classes)
    
    print(feats_samples)
    
    print('shape:',feats_samples[0].shape)
    
    print('Calculating TSNE:')
    
    model = TSNE(n_components=2, random_state=0, metric='euclidean')
    tsne_data = model.fit_transform(np.vstack(feats_samples))
    
    #model = TSNE(n_components=2, random_state=0, metric='cosine')
    #tsne_data = model.fit_transform([feat[100:] for feat in feats])

    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
    colorst = colormap(np.linspace(0, 0.9, num_classes+1)) #[colormap(i) for i in np.linspace(0, 0.9, num_speakers)]  
       
    cs = [colorst[feats_samples_classes_num[i]] for i in range(len(feats_samples_classes_num))]
    
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

    
def visualize_kaldi_bin_feats(feat_filename, max_frames, num_feat=0, phn_file='', phn_offset=5, wav_file='', do_tsne=False):
    feats, utt_ids = kaldi_io.readArk(feat_filename , limit=10000)
    
    print([feat.shape for feat in feats], utt_ids)
    
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
                        'feats/feats_sp_transVgg16big_nsampling_rnd_win64_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0005_featinput_unnormalized.feats.ark_dot_combine_tied_embs/dev/feats.ark')
                        #'feats_vgg.ark')
    parser.add_argument('-f', '--format', dest='format', help='Format of the feature file (raw,kaldi_ark)', type=str, default = 'kaldi_ark')
    parser.add_argument('-m', '--max_frames', dest='max_frames', help='Maximum frames', type=int, default = 10000)
    parser.add_argument('-n', '--num_feat', dest='num_feat', help='feat file to visualize', type=int, default = 2)
    parser.add_argument('-p', '--phn_file', dest='phn_file', help='Phoneme annotation file', type=str, default = '')
    parser.add_argument('-u', '--utt_2_class', dest='utt_2_class', help='File with meta classes, e.g. male / female / age etc.', type=str, default = '')
    
    parser.add_argument('--mode', dest='mode', help='(featshow|stats|classes_tsne)', type=str, default = 'featshow')


    args = parser.parse_args()
    
    if args.mode=='stats':
        visualize_stats(args.featfile, args.max_frames)
    elif args.mode=='featshow':
        visualize_kaldi_bin_feats(args.featfile, args.max_frames, phn_file= args.phn_file, num_feat=args.num_feat)
    elif args.mode=='classes_tsne':
        visualize_classes_tsne(args.featfile, utt_2_class_filename=args.utt_2_class, half_index=100)
    else:
        print("mode not supported.")
        #visualize_stats(args.featfile, args.max_frames)
    
    #if args.format == 'kaldi_ark':
    #    
