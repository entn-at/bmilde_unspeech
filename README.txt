Unspeech training code for “Unsupervised Speech Context Embeddings”

If you use our code or models in your academic work, please cite this paper:

“Unspeech: Unsupervised Speech Context Embeddings”, Benjamin Milde, Chris Biemann, Proceedings of Interspeech 2018, Hyderabad, India

Visit http://unspeech.net for more information, examples on training models, using them to generate features and clustering them. There are also pretrained models available for some of the models that were evaluated in our paper.

Short overview of the main programs:

unsup_model_neg.py – Main training and feature generation code, using a discriminative objective function. Works with Tensorflow 1.5+, tested with 1.8.
unsup_model.py – Some first experiments with other objective functions and a generative model of speech. Not used in the paper, Pre-Tensorflow 1.0 code.
unsup_model_10.py – Similar to unsup_model.py but updated to Tensorflow 1.0 (will not work with newer versions)
show_feats.py – can be used to visualize features in Kaldi ark,scp format (FBANK, MFCC, unspeech…)
cluster.py  - cluster features with HDBSCAN, evaluate with ARI / NMI also visualize clusters with TSNE.
