import wave
import numpy as np
import scipy
import os
import scipy.io.wavfile

#compresses the dynamic range, see https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
def encode_mulaw(signal,mu=255):
    return np.sign(signal)*(np.log(1.0+mu*np.abs(signal)) / np.log(1.0+mu))

#uncompress the dynamic range, see https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
def decode_mulaw(signal,mu=255):
    return np.sign(signal)*(1.0/mu)*(np.power(1+mu,np.abs(signal))-1.0)

def readWordPosFile(filename,pos1=0,pos2=1):
    unalign_list = []
    with open(filename) as f:
        for line in f.readlines():
            split = line[:-1].split(" ")
            unalign_list.append((float(split[pos1]), float(split[pos2])))
    return unalign_list

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def loadIdFile(idfile,use_no_files=-1):
    ids = []
    with open(idfile) as f:
        ids = f.read().split('\n')[:use_no_files]
    #check if ids exist
    #ids = [myid for myid in ids if os.path.ispath(myid)]
    return [myid for myid in ids if myid != '']

def getSignal(utterance):
    spf = wave.open(utterance, 'r')
    sound_info = spf.readframes(-1)
    signal = np.fromstring(sound_info, 'Int16')
    return signal, spf.getframerate()

def writeSignal(signal, myfile, rate=16000, do_decode_mulaw=False):
    if do_decode_mulaw:
        signal = decode_mulaw(signal)
    return scipy.io.wavfile.write(myfile, rate, signal)

def rolling_window(a, window_len, hop):
    shape = a.shape[:-1] + (a.shape[-1] - window_len + 1, window_len)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::hop]

def writeArkTextFeatFile(feat, feat_name, out_filename, append = False):
    with open(out_filename, 'a' if append else 'w') as out_file:
        out_file.write(feat_name  + ' [')
        for feat_vec in feat:
            feat_vec_str = ' '.join([str(elem) for elem in feat_vec])
            out_file.write(feat_vec_str)
    
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def writeZeroSpeechFeatFile(feat, out_filename, window_length, hop_size):
    ensure_dir(out_filename)
    with open(out_filename, 'w') as out_file:
        for i,feat_vec in enumerate(feat):
            pos = i * hop_size + (window_length / 2.0)
            feat_vec_str = ' '.join([str(elem) for elem in feat_vec])
            out_file.write(str(pos) + ' ' + feat_vec_str + '\n')
