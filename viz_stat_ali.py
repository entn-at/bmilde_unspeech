#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 18:24:39 2017

@author: me
"""

# Calculate statistics about Kaldi alignments in text format. After training, to obtain them, e.g. do the following:
# cd ~/kaldi/egs/tedlium/s5_r2/exp/tri3_ali_cleaned
# gunzip -c ali.*.gz | show-alignments ../../data/lang/phones.txt final.mdl ark:- > ali.txt

import numpy as np
import scipy

from scipy import stats

from collections import defaultdict
import matplotlib.pyplot as plt


import matplotlib
matplotlib.rcParams.update({'font.size': 12})

from matplotlib.ticker import AutoMinorLocator

filename='ali.txt'

phn_len_dict = defaultdict(list)

with_phn_end_markers = True 

with open(filename) as ali:
    mode = 'read_states'
    elem_list = []
    line_num=0
    print('opened',filename)
    for line in ali:
        if line.strip() == '':
            continue
        if mode == 'read_states':
            count = 0
            do_count=False
            for elem in line.split():
                if do_count:
                    count += 1
                if elem == '[':
                    do_count=True
                    count=0
                if elem == ']':
                    do_count=False
                    elem_list.append(count)
                    count=0
                
        elif mode == 'assign_states':
            if line[-1]=='\n':
                line=line[:-1]
            split = line.split()
            split = split[1:]   
            
            assert(len(split) == len(elem_list))
            
            for phn,length in zip(split, elem_list):
                if not with_phn_end_markers:
                    phn = phn.split('_')[0]
                phn_len_dict[phn] += [length]
                
            elem_list = []
            
        mode = ('assign_states' if mode=='read_states' else 'read_states')

       # print(line_num,mode)
        
        line_num += 1
        
#        if line_num > 1000:
#            break

phn_len_list = [(phn,np.asarray(n)) for phn,n in phn_len_dict.items()]

phn_avg_list = [(phn,np.mean(array)) for phn,array in phn_len_list]

avgs = np.concatenate([array for phn,array in phn_len_list], axis=0)
                    
print('mean of all phonemes is:', np.mean(avgs))

phn_avg_list.sort(key=lambda x: x[1], reverse=True)

phn_len_list = [(phn,np.asarray(phn_len_dict[phn])) for phn,avg in phn_avg_list]

print(phn_len_list)

phn_min_list = [np.min(array) for phn,array in phn_len_list]
phn_max_list = [np.max(array) for phn,array in phn_len_list]
phn_mean_list = [np.mean(array) for phn,array in phn_len_list]
 
phn_count_list = [len(array) for phn,array in phn_len_list]

print('phn_count_list:',phn_count_list)

print(phn_avg_list)
print(phn_min_list)
print(phn_max_list)

phn_num_list = list(range(len(phn_len_list)))

phn_num_list_sum = sum(phn_count_list)

print("There are ", str(len(phn_len_list)), "phonemes.")
print("Counted:", phn_num_list_sum, "phones.")

phn_num_list = [float(x) for x in phn_num_list]


plt.errorbar(phn_num_list, phn_mean_list, yerr=[np.std(n) for phn,n in phn_len_list]) 
             #yerr=[stats.sem(n) for phn,n in phn_len_list])    #[phn_min_list,phn_max_list])
plt.xticks(phn_num_list, [x[0] for x in phn_len_list], rotation='vertical')

plt.yticks(range(0, 50,5))

#plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
#plt.xtickslabels( [x[0] for x in phn_len_list], rotation='vertical')

minorLocator = AutoMinorLocator()
plt.gca().yaxis.set_minor_locator(minorLocator)


plt.xlabel('phone')
plt.ylabel('duration (mean/std no. frames)')

plt.show()
       
#ax.errorbar(x, y, yerr=[yerr, 2*yerr], xerr=[xerr, 2*xerr], fmt='--o')
                