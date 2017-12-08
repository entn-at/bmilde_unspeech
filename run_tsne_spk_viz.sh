run=/Users/milde/inspect/1508243155feats_transVgg16_nsampling_rnd_win50_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0005_dot_combine

#run=
num_filters=40
embedding_transformation=Vgg16
output_feat_format=kaldi_bin
num_highway_layers=5
num_dnn_layers=5
embedding_size=100
hop_size=1
#additional_params="--batch_normalization --fc_size 1024 --unit_normalize_var"
additional_params="--fc_size 512 --unit_normalize_var --window_length 50 --window_neg_length 50"

echo "computing feats for dev set..."
python3 unsup_model_neg.py --tnse_viz_speakers --spk2utt data/dev/spk2utt --train_dir $run  --filelist data/dev/unnormalized.feats.ark  --num_filters $num_filters --embedding_transformation $embedding_transformation --num_highway_layers $num_highway_layers --embedding_size $embedding_size --num_dnn_layers $num_dnn_layers --hop_size $hop_size --additional_params $additional_params
