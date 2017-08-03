import wave
import numpy as np
import scipy
import os
import scipy.io.wavfile
import tensorflow as tf
import os.path
import gzip
import bz2
import wavefile

def smart_open(filename, mode = 'rb', *args, **kwargs):
    '''
    Opens a file "smartly":
      * If the filename has a ".gz" or ".bz2" extension, compression is handled
        automatically;
      * If the file is to be read and does not exist, corresponding files with
        a ".gz" or ".bz2" extension will be attempted.
    '''
    readers = {'.gz': gzip.GzipFile, '.bz2': bz2.BZ2File}
    if 'r' in mode and not os.path.exists(filename):
        for ext in readers:
            if os.path.exists(filename + ext):
                filename += ext
                break
    extension = os.path.splitext(filename)[1]
    return readers.get(extension, open)(filename, mode, *args, **kwargs)

#compresses the dynamic range, see https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
def encode_mulaw(signal,mu=255):
    return np.sign(signal)*(np.log1p(mu*np.abs(signal)) / np.log1p(mu))

#uncompress the dynamic range, see https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
def decode_mulaw(signal,mu=255):
    return np.sign(signal)*(1.0/mu)*(np.power(1.0+mu,np.abs(signal))-1.0)

# discretize signal between -1.0 and 1.0 into mu+1 bands.
def discretize(signal, mu=255.0):
    output = np.array(signal)
    output += 1.0
    output = output*(0.5*mu)
    signal = np.fmax(0.0,output)
    #signal = np.fmin(255.0,signal)
    return signal.astype(np.int32)

def undiscretize(signal, mu=255.0):
    output = np.array(signal)
    output = output.astype(np.float32)
    output /= 0.5*mu
    output -= 1.0
    signal = np.fmax(-1.0,output)
    signal = np.fmin(1.0,signal)
    return signal

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
    
    ids = [myid for myid in ids if myid != '']
    
    if len(ids[0].split()) > 1:
        utt_ids = []
        wav_files = []
        
        for myid in ids:
            print(myid)
            split = myid.split()
            utt_ids.append(split[0])
            wav_files.append(split[1])
    else:
        utt_ids = []
        wav_files = ids
    #check if ids exist
    #ids = [myid for myid in ids if os.path.ispath(myid)]
    return utt_ids, wav_files

def loadPhnFile(phn_file):
    positions = []
    names = []
    with open(phn_file) as phn:
        for line in phn:
            if line[-1] == '\n':
                line = line[:-1]
            split = line.split()
            pos = (split[0],split[1])
            name = split[-1]
            positions.append(pos)
            names.append(name)
    return positions,names

def loadUtt2Spk(utt_filename):
    utts = {}
    with open(utt_filename) as utt_file:
        for line in utt_file:
            if line[-1] == '\n':
                line = line[:-1]
            split = line.split()
            utt = split[0]
            spk = split[1]
            utts[utt] = spk
    return utts

def getSignalOld(utterance):
    spf = wave.open(utterance, 'r')
    sound_info = spf.readframes(-1)
    signal = np.fromstring(sound_info, 'Int16')
    return signal, spf.getframerate()

# This is needed since the old loader had problems with NIST headers from TIMIT. 
# See also https://stackoverflow.com/questions/10187043/read-nist-wav-file-in-timit-database-into-python-numpy-array
def getSignal(utterance):
    samplerate, signal = wavefile.load(utterance)
    print(signal)
    signal = signal[0]
    #print(utterance, 'dtype:', signal.dtype, 'min:', min(signal), 'max:', max(signal), 'samplerate:', samplerate)
    return signal, samplerate

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

def writeZeroSpeechFeatFile(feat, out_filename, window_length, hop_size):
    ensure_dir(out_filename)
    with open(out_filename, 'w') as out_file:
        for i,feat_vec in enumerate(feat):
            pos = i * hop_size + (window_length / 2.0)
            feat_vec_str = ' '.join([str(elem) for elem in feat_vec])
            out_file.write(str(pos) + ' ' + feat_vec_str + '\n')
            
def tensor_normalize_0_to_1(in_tensor):
    x_min = tf.reduce_min(in_tensor)
    x_max = tf.reduce_max(in_tensor)
    tensor_0_to_1 = ((in_tensor - x_min) / (x_max - x_min))
    return tensor_0_to_1
