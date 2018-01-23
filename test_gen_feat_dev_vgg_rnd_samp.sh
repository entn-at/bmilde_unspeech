run=models/1511893200feats_transResnet_v2_50_small_nsampling_same_spk_win16_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0001_dot_combine  #/Users/milde/inspect/1508243155feats_transVgg16_nsampling_rnd_win50_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0005_dot_combine
run=models/1512519850feats_transResnet_v2_50_small_nsampling_same_spk_win32_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_dropout_keep0.9_l2_reg0.0001_dot_combine

#run=models/1512051714feats_transVgg16big_nsampling_same_spk_win32_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0001_dot_combine

run=models/1512051734feats_transVgg16big_nsampling_rnd_win32_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0001_dot_combine
run=models/1515600875feats_transVgg16big_nsampling_rnd_win64_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0001_featinput_unnormalized.feats.ark_dot_combine_tied_embs

num_filters=40
embedding_transformation=Vgg16big
output_feat_format=kaldi_bin
num_highway_layers=5
num_dnn_layers=5
embedding_size=100
hop_size=1
#additional_params="--batch_normalization --fc_size 1024 --unit_normalize_var"
#additional_params="--fc_size 512 --unit_normalize_var --window_length 16 --window_neg_length 16 --unit_normalize_var"
additional_params="--fc_size 512 --window_length 64 --window_neg_length 64 --unit_normalize_var --tied_embeddings_transforms --nogenerate_speaker_vectors --genfeat_combine_contexts --nokaldi_normalize_to_input_length --notest_perf"

echo "computing feats for dev set..."
python3 unsup_model_neg.py --gen_feat --spk2utt fake --train_dir $run  --filelist feats/tedlium/dev/unnormalized.feats.ark  --num_filters $num_filters --embedding_transformation $embedding_transformation --num_highway_layers $num_highway_layers --embedding_size $embedding_size --num_dnn_layers $num_dnn_layers --hop_size $hop_size --additional_params $additional_params --output_feat_file feats/%model_params/dev/feats

echo "computing feats for test set... "
python3 unsup_model_neg.py --gen_feat --spk2utt fake --train_dir $run  --filelist feats/tedlium/test/unnormalized.feats.ark  --num_filters $num_filters --embedding_transformation $embedding_transformation --num_highway_layers $num_highway_layers --embedding_size $embedding_size --num_dnn_layers $num_dnn_layers --hop_size $hop_size --additional_params $additional_params --output_feat_file feats/%model_params/test/feats

echo "computing feats for train set... "
python3 unsup_model_neg.py --gen_feat --spk2utt fake --train_dir $run  --filelist feats/tedlium/train/unnormalized.feats.ark  --num_filters $num_filters --embedding_transformation $embedding_transformation --num_highway_layers $num_highway_layers --embedding_size $embedding_size --num_dnn_layers $num_dnn_layers --hop_size $hop_size --additional_params $additional_params --output_feat_file feats/%model_params/train/feats
