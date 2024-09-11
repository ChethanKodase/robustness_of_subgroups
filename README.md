# Robustness of subgroups
This code is part of the paper Evaluating Adversarial Robustness of Variational Autoencoders across intersectional subgroups

This repository contains code and material for the article [Evaluating Adversarial Robustness of Variational Autoencoders across intersectional subgroups](https://arxiv.org/abs/2407.03864v1)

## Datasets
* CelebA : [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) 

## Autoencoder model: 
* Variational Autoencoder

## VAE training :

* Put all the training images of celebA dataset in `./data_faces/image_directory/` in the home directory

* cd into the home directory and run `python train_aautoencoders/vae_celebA_training.py` 
* The model can be accesses from the directory `./train_aautoencoders/saved_model/checkpoints` after trainig.

## Generating data subgroups
* Use the jupyter notebook `celebA_grouping` to group the dataset into `Young`, `Old`, `Men`, `Women`, `Young men`, `Young women`, `Old men` and `Old women`.


## Generating maximum damage attack to evaluate adversarial robustness 

Perturbations for maximum damage attack on individual samples can be generated by running `python attck_L_inf_bounded.py --feature_no 0 --sample_no 0 --which_gpu 0`

where `feature_no` represents the index of the group in the list `all_features = [ "men", "women", "young", "old", "youngmen", "oldmen", "youngwomen", "oldwomen"]`
from which you want to sample an image to attack and `sample_no` indicates the index of the sample to be selected from each group which consisted of 60 samples as mentioned in the paper. Assign value to `which_gpu` based on the gpu you want to use for the optimization. 


![plot](./qualitative_beta1.png)

Inputs and reconstructions for normal and perturbed samples from the groups `young men` (columns 1 & 2), `young women` (columns 3 & 4), `old men` (columns 5 & 6),  and `old women` (columns 7 & 8) for Vanilla-VAE with $\beta=1.0$
