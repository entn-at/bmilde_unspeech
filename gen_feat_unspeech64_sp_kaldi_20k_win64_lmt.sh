run=models/1511893200feats_transResnet_v2_50_small_nsampling_same_spk_win16_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0001_dot_combine  #/Users/milde/inspect/1508243155feats_transVgg16_nsampling_rnd_win50_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0005_dot_combine
run=models/1512519850feats_transResnet_v2_50_small_nsampling_same_spk_win32_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_dropout_keep0.9_l2_reg0.0001_dot_combine

#run=models/1512051714feats_transVgg16big_nsampling_same_spk_win32_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0001_dot_combine

run=models/1512051734feats_transVgg16big_nsampling_rnd_win32_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0001_dot_combine
run=models/1515600875feats_transVgg16big_nsampling_rnd_win64_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0001_featinput_unnormalized.feats.ark_dot_combine_tied_embs

# sp, trained longer
run=models/1519252232feats_transVgg16big_nsampling_rnd_win32_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0001_featinput_feats_unnormalized.ark_dot_combine_tied_embs

# sp, trained longer:
run=models/1519320339feats_transVgg16big_nsampling_rnd_win64_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0001_featinput_feats_unnormalized.ark_dot_combine_tied_embs

#big win128, trained 60h
run=models/1520559357feats_transVgg16big_nsampling_rnd_win128_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size2048_unit_norm_var_dropout_keep0.9_l2_reg0.0001_featinput_filelist.english.train_dot_combine_tied_embs_mt

#big win64, trained 160h
run=models/1520559132feats_transVgg16big_nsampling_rnd_win64_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size2048_unit_norm_var_dropout_keep0.9_l2_reg0.0001_featinput_filelist.english.train_dot_combine_tied_embs_lmt

num_filters=40
embedding_transformation=Vgg16big
output_feat_format=kaldi_bin
num_highway_layers=5
num_dnn_layers=5
embedding_size=100
hop_size=1
genfeat_stride=10
#additional_params="--batch_normalization --fc_size 1024 --unit_normalize_var"
#additional_params="--fc_size 512 --unit_normalize_var --window_length 16 --window_neg_length 16 --unit_normalize_var"
additional_params="--fc_size 2048 --window_length 64 --window_neg_length 64 --unit_normalize_var --tied_embeddings_transforms --nogenerate_speaker_vectors --nogenfeat_combine_contexts --nokaldi_normalize_to_input_length --genfeat_stride $genfeat_stride --notest_perf --genfeat_interpolate_outputlength_padding"

outputdir=/srv/data/milde/kaldi/egs/tedlium/s5_r2/data/unspeech_64_20k_lmt

echo "computing feats for dev set..."
#python3 unsup_model_neg.py --gen_feat --train_dir $run  --filelist /srv/data/milde/kaldi/egs/tedlium/s5_r2/data/dev_fbank/unnormalized.feats.ark  --num_filters $num_filters --embedding_transformation $embedding_transformation --num_highway_layers $num_highway_layers --embedding_size $embedding_size --num_dnn_layers $num_dnn_layers --hop_size $hop_size --additional_params $additional_params --output_feat_file $outputdir/dev/ivector_online

echo $genfeat_stride > $outputdir/dev/ivector_period

echo "computing feats for test set... "
#python3 unsup_model_neg.py --gen_feat --train_dir $run  --filelist /srv/data/milde/kaldi/egs/tedlium/s5_r2/data/test_fbank/unnormalized.feats.ark  --num_filters $num_filters --embedding_transformation $embedding_transformation --num_highway_layers $num_highway_layers --embedding_size $embedding_size --num_dnn_layers $num_dnn_layers --hop_size $hop_size --additional_params $additional_params --output_feat_file $outputdir/test/ivector_online

echo $genfeat_stride > $outputdir/test/ivector_period

echo "computing feats for train set... "
#python3 unsup_model_neg.py --gen_feat --train_dir $run  --filelist /srv/data/milde/kaldi/egs/tedlium/s5_r2/data/train_cleaned_sp_hires_fbank_comb/unnormalized.feats.ark  --num_filters $num_filters --embedding_transformation $embedding_transformation --num_highway_layers $num_highway_layers --embedding_size $embedding_size --num_dnn_layers $num_dnn_layers --hop_size $hop_size --additional_params $additional_params --output_feat_file $outputdir/train_cleaned_sp_comb/ivector_online

echo $genfeat_stride > $outputdir/train_cleaned_sp_comb/ivector_period

echo "computing feats for commonvoice dev set..."
python3 unsup_model_neg.py --gen_feat --train_dir $run  --filelist  /srv/data/milde/kaldi/egs/tedlium/s5_r2/data/valid_dev_fbank/feats.unnormalized.ark --num_filters $num_filters --embedding_transformation $embedding_transformation --num_highway_layers $num_highway_layers --embedding_size $embedding_size --num_dnn_layers $num_dnn_layers --hop_size $hop_size --additional_params $additional_params --output_feat_file $outputdir/valid_dev/ivector_online

echo $genfeat_stride > $outputdir/valid_dev/ivector_period

echo "computing feats for commonvoice test set... "
python3 unsup_model_neg.py --gen_feat --train_dir $run  --filelist  /srv/data/milde/kaldi/egs/tedlium/s5_r2/data/valid_test_fbank/feats.unnormalized.ark --num_filters $num_filters --embedding_transformation $embedding_transformation --num_highway_layers $num_highway_layers --embedding_size $embedding_size --num_dnn_layers $num_dnn_layers --hop_size $hop_size --additional_params $additional_params --output_feat_file $outputdir/valid_test/ivector_online

echo $genfeat_stride > $outputdir/valid_test/ivector_period
