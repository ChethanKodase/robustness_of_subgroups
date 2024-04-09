# robustness_of_subgroups
Evaluating Adversarial Robustness of Variational Autoencoders across intersectional subgroups

This repository contains code and material for the article "Evaluating Adversarial Robustness of Variational Autoencoders across intersectional subgroups"

## Datasets
* CelebA : [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) 

## Autoencoder model: 
* Variational Autoencoder

## VAE training :

* Put all the training images of celebA dataset in `./data_faces/image_directory/` in the home directory

* cd into the home directory and run `python train_aautoencoders/vae_celebA_training.py` 
* The model can be accesses from the directory `./train_aautoencoders/saved_model/checkpoints` after trainig.