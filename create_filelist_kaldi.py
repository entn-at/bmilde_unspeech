#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:30:36 2017

@author: milde
"""

import argparse
import io

def create_filelist(input_file, output_file):
    with io.open(input_file, 'r') as input_file, io.open(output_file, 'w') as output_file:
        for line in input_file:
            split = line.split()
            for elem in split:
                if ".wav" in elem:
                    output_file.write(split[0] +' '+ elem + '\n')
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts a Kaldi scp into a simpler filelist format')
    parser.add_argument('-i', '--input_scp', dest='input', help='The input file.', type=str, default = '')
    parser.add_argument('-o', '--output_filelist', dest='out', help='The output filelist file.', type=str, default = '')
    
    args = parser.parse_args()
    
    create_filelist(args.input, args.out)
     
     

    