#!/bin/bash

for set in dev test;
do
	for feats in "feats/tedlium_ivectors_sp/%set/ivector_online.ark" \
		     "feats/tedlium_ivectors_nosp/%set/ivector_online.ark" \
		     "feats/feats_sp_transVgg16big_nsampling_rnd_win32_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0005_featinput_unnormalized.feats.ark_dot_combine_tied_embs/%set/feats.ark" \
		     "feats/feats_nosp_transVgg16big_nsampling_rnd_win64_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0005_featinput_unnormalized.feats.ark_dot_combine_tied_embs/%set/feats.ark" \
		     "feats/feats_sp_transVgg16big_24h_nsampling_rnd_win64_neg_samples4_lcontexts2_rcontexts2_flts40_embsize100_fc_size512_unit_norm_var_dropout_keep0.9_l2_reg0.0005_featinput_unnormalized.feats.ark_dot_combine_tied_embs/%set/feats.ark"		
	do
		if [[ $feats = *"ivector"* ]]; then
			python3 cluster.py --input-ark $feats --set $set --half_index -1
		else
			python3 cluster.py --input-ark $feats --set $set --half_index 100
		fi
		
	done
done
