# no speaker information
CUDA_VISIBLE_DEVICES=3 python3 unsup_model_neg.py --window_length 16 --window_neg_length 16 --filelist /srv/data/milde/eesen/asr_egs/tedlium/v2-30ms/data/train/unnormalized.feats.ark --noend_to_end --embedding_transform Resnet_v2_50_small --unit_normalize_var True --l2_reg 0.0001 --batch_size 32
