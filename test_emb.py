from io import StringIO
import numpy as np
from collections import defaultdict

emb='data/word_embs/tedlium_word_emb_ted_spks_1.emb'
embds_dict = defaultdict(list)

out_file_tsv=emb[:-4]+'.tsv'
out_file_labels=emb[:-4]+'.labels'

i=0

with open(emb) as emb_file:
    for line in emb_file:
        if i%1000 == 0:
            print('At word:',i)
        split = line.split()
        myid = split[0]
        word = split[1]
        emb_str = ' '.join(split[2:])
        #emb_str = emb_str.replace(' ','').replace(',',', ')
        emb = np.genfromtxt(StringIO(emb_str[1:-1]), dtype=np.float32, delimiter=',')
#        print(word, emb)
        embds_dict[word] += [emb]
        i+=1

#        if i>10000:
#            break

with open(out_file_tsv,'w') as out_file_tsv_file:
    with open(out_file_labels,'w') as out_file_labels_file:
        for word in embds_dict:
            emb_list = embds_dict[word] 
            emb = np.mean(emb_list,axis=0)
            out_file_labels_file.write(word+'\n')
            emb_str='\t'.join([str(elem) for elem in emb])
            out_file_tsv_file.write(emb_str+'\n')
            #print(word, np.mean(emb_list,axis=0))
