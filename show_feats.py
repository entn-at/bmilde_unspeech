#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:26:30 2017

@author: me
"""

import pylab as  plt
import numpy as np

feat = "1009.fea"
max_frames = 100

i = 0
feats = []
with open(feat) as feat_in:
    for line in feat_in:
        i += 1
        if line[-1]=='\n':
            line = line[:-1]
        frame = np.asarray([float(elem) for elem in line.split(' ')])[1:]
        print(frame)
        if i == max_frames:
            break
        
plt.matshow(feats)
plt.show()