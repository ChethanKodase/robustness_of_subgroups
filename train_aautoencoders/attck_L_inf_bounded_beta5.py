import torch

import torch.nn as nn

#from nvae.utils import add_sn
#from nvae.vae_celeba import NVAE
import numpy as np
import matplotlib.pyplot as plt
#from nvae.utils import reparameterize




'''

cd /home/luser/robustness_of_subgroups/train_aautoencoders
conda activate inn
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 40 --which_gpu 0
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 41 --which_gpu 0
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 42 --which_gpu 0
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 43 --which_gpu 0
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 44 --which_gpu 0

cd /home/luser/robustness_of_subgroups/train_aautoencoders
conda activate inn
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 45 --which_gpu 0
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 46 --which_gpu 0
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 47 --which_gpu 0
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 48 --which_gpu 0
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 49 --which_gpu 0

cd /home/luser/autoencoder_attacks/train_aautoencoders/
conda activate inn
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 50 --which_gpu 0
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 51 --which_gpu 0
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 52 --which_gpu 0
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 53 --which_gpu 0
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 54 --which_gpu 0

cd /home/luser/autoencoder_attacks/train_aautoencoders/
conda activate inn
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 55 --which_gpu 0
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 56 --which_gpu 0
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 57 --which_gpu 0
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 58 --which_gpu 0
python attck_L_inf_bounded_beta5.py --feature_no 12 --segment 59 --which_gpu 0
######################################################################################################################################


cd /home/luser/autoencoder_attacks/train_aautoencoders/
conda activate inn
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 40 --which_gpu 1
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 41 --which_gpu 1
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 42 --which_gpu 1
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 43 --which_gpu 1
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 44 --which_gpu 1

cd /home/luser/autoencoder_attacks/train_aautoencoders/
conda activate inn
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 45 --which_gpu 1
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 46 --which_gpu 1
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 47 --which_gpu 1
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 48 --which_gpu 1
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 49 --which_gpu 1

cd /home/luser/autoencoder_attacks/train_aautoencoders/
conda activate inn
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 50 --which_gpu 1
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 51 --which_gpu 1
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 52 --which_gpu 1
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 53 --which_gpu 1
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 54 --which_gpu 1

cd /home/luser/autoencoder_attacks/train_aautoencoders/
conda activate inn
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 55 --which_gpu 1
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 56 --which_gpu 1
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 57 --which_gpu 1
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 58 --which_gpu 1
python attck_L_inf_bounded_beta5.py --feature_no 13 --segment 59 --which_gpu 1
######################################################################################################################################

cd /home/luser/autoencoder_attacks/train_aautoencoders/
conda activate inn
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 40 --which_gpu 2
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 41 --which_gpu 2
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 42 --which_gpu 2
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 43 --which_gpu 2
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 44 --which_gpu 2


cd /home/luser/autoencoder_attacks/train_aautoencoders/
conda activate inn
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 45 --which_gpu 2
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 46 --which_gpu 2
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 47 --which_gpu 2
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 48 --which_gpu 2
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 49 --which_gpu 2

cd /home/luser/autoencoder_attacks/train_aautoencoders/
conda activate inn
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 50 --which_gpu 2
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 51 --which_gpu 2
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 52 --which_gpu 2
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 53 --which_gpu 2
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 54 --which_gpu 2



cd /home/luser/autoencoder_attacks/train_aautoencoders/
conda activate inn
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 55 --which_gpu 2
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 56 --which_gpu 2
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 57 --which_gpu 2
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 58 --which_gpu 2
python attck_L_inf_bounded_beta5.py --feature_no 14 --segment 59 --which_gpu 2

######################################################################################################################################

cd /home/luser/autoencoder_attacks/train_aautoencoders/
conda activate inn
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 40 --which_gpu 3
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 41 --which_gpu 3
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 42 --which_gpu 3
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 43 --which_gpu 3
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 44 --which_gpu 3

cd /home/luser/autoencoder_attacks/train_aautoencoders/
conda activate inn
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 45 --which_gpu 3
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 46 --which_gpu 3
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 47 --which_gpu 3
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 48 --which_gpu 3
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 49 --which_gpu 3

cd /home/luser/autoencoder_attacks/train_aautoencoders/
conda activate inn
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 50 --which_gpu 3
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 51 --which_gpu 3
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 52 --which_gpu 3
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 53 --which_gpu 3
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 54 --which_gpu 3


cd /home/luser/autoencoder_attacks/train_aautoencoders/
conda activate inn
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 55 --which_gpu 3
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 56 --which_gpu 3
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 57 --which_gpu 3
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 58 --which_gpu 3
python attck_L_inf_bounded_beta5.py --feature_no 15 --segment 59 --which_gpu 3


'''


#device = "cuda:1" if torch.cuda.is_available() else "cpu"

import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VAE')
parser.add_argument('--feature_no', type=int, default=5, help='Index of the feature to attack (0-5)')
parser.add_argument('--segment', type=int, default=3, help='Segment index')
parser.add_argument('--which_gpu', type=int, default=3, help='Index of the GPU to use (0-N)')
args = parser.parse_args()

feature_no = args.feature_no
segment = args.segment
which_gpu = args.which_gpu


device = ("cuda:"+str(which_gpu)+"" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training

from vae import VAE_big

model = VAE_big(device, image_channels=3).to(device)

train_data_size = 162079
epochs = 199

beta_value = 10.0

#model.load_state_dict(torch.load('/home/luser/autoencoder_attacks/saved_celebA/checkpoints/celebA_CNN_VAE_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))
model.load_state_dict(torch.load('/home/luser/autoencoder_attacks/saved_celebA/checkpoints/celebA_CNN_VAE'+str(beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))
model.eval()

roll_no = 1

#segment = 20


all_features = ["bald", "beard", "oldfemaleGlass", "hat", "blackWomen", "generalWhiteWomen", "blackMen", "generalWhiteMen", "men", "women", "young", "old", "youngmen", "oldmen", "youngwomen", "oldwomen", "oldblackmen", "oldblackwomen", "oldwhitemen", "oldwhitewomen", "youndblackmen", "youndblackwomen", "youngwhitemen", "youngwhitewomen" ]

populations_all_features = ["bald", "beard", "oldfemaleGlass", "hat", "blackWomen", "generalWhiteWomen", "blackMen", "generalWhiteMen", "men : 84434", "women : 118165", "young : 156734", "old : 45865", "youngmen : 53448 ", "oldmen : 7003", "youngwomen : 103287", "oldwomen : 1116" ]

select_feature = all_features[feature_no]


source_im = torch.load("/home/luser/autoencoder_attacks/train_aautoencoders/fairness_trials/attack_saves/"+str(select_feature)+"_d/images.pt")[segment].unsqueeze(0).to(device) 
model.eval()

print("source_im.shape", source_im.shape)


noise_addition = 2.0 * torch.rand(1, 3, 64, 64).to(device) - 1.0

#noise_addition = 0.08 * (2 * noise_addition - 1)
torch.norm(noise_addition, 2)


#desired_norm_l2 = 6.0  # Change this to your desired constant value
desired_norm_l_inf = 0.094  # Worked very well

#desired_norm_l_inf = 0.04  # Worked very well


import torch.optim as optim
optimizer = optim.Adam([noise_addition], lr=0.0001)
adv_alpha = 0.5

noise_addition.requires_grad = True

criterion = nn.MSELoss()

num_steps = 250000

prev_loss = 0.0

for step in range(num_steps):

    #current_L_2_norm = torch.norm(noise_addition, 2)
    current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))

    scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 

    attacked = (source_im + scaled_noise)

    normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)


    '''mu1, log_var1, xs1 = model.encoder(normalized_attacked)
    z1 = reparameterize(mu1, torch.exp(0.5 * log_var1))'''

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1



    '''mu2, log_var2, xs2 = model.encoder(source_im.to(device))
    z2 = reparameterize(mu2, torch.exp(0.5 * log_var2))'''

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2



    lipschitzt_loss_encoder = criterion(z1, z2) / ( criterion(normalized_attacked, normalized_attacked) ) 

    lipschitzt_loss_decoder = ( criterion(normalized_attacked, adv_gen) ) / criterion(z1, z2) 

    l2_distortion = torch.norm(scaled_noise, 2)
    l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))


    #total_loss = (-1 * (criterion(z1, z2))  ) * (1-adv_alpha)  + adv_alpha  * (  (  ( distortion - distort_valuae)**2  )  +  1e-9  )**(1/2)
    total_loss = (-1 * (criterion(z1, z2))  )   #+ adv_alpha  * (  (  ( distortion - distort_valuae)**2  )  +  1e-9  )**(1/2)

    total_loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    # Print the loss every 100 steps
    if step % 10000 == 0:
        convergence_value = ((prev_loss - total_loss.item())**2)**(0.5)
        print(f"Step {step}, Loss: {total_loss.item()}, distortion L-2: {l2_distortion}, distortion L-inf: {l_inf_distortion}")
        prev_loss = total_loss.item()
        '''with torch.no_grad():
            fig, ax = plt.subplots(1, 3, figsize=(10, 10))
            ax[0].imshow(normalized_attacked[0].permute(1, 2, 0).cpu().numpy())
            ax[0].set_title('Attacked Image')
            ax[0].axis('off')

            ax[1].imshow(scaled_noise[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
            ax[1].set_title('Noise')
            ax[1].axis('off')

            ax[2].imshow(adv_gen[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
            ax[2].set_title('Attack reconstruction')
            ax[2].axis('off')
            plt.show()
            plt.savefig("/home/luser/autoencoder_attacks/train_aautoencoders/fairness_trials/attack_saves/"+str(select_feature)+"_d/TEST_scaled_step"+str(step)+"attack_"+str(desired_norm_l_inf)+"segment"+str(segment)+".png")'''

        optimized_noise = scaled_noise
        torch.save(optimized_noise, "/home/luser/autoencoder_attacks/train_aautoencoders/fairness_trials/attack_saves/"+str(select_feature)+"_d/"+str(select_feature)+"beta"+str(beta_value)+"_scaled_noise_"+str(desired_norm_l_inf)+"segment"+str(segment)+".pt")
        