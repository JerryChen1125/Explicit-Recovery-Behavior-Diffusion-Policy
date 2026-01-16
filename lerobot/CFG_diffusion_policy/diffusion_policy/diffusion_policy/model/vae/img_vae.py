import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from diffusion_policy.common.pytorch_util import dict_apply
class ImageSeqVAE(nn.Module):
    def __init__(self,shape_meta,horizon,obs_encoder, latent_dim):
        super(ImageSeqVAE, self).__init__()
        self.img_shape=[3,96,96]
        self.img_channels=3
        self.hidden_dim=latent_dim
        self.latent_dim=514
        self.horizon=horizon
        self.obs_encoder=obs_encoder
        self.out_dim=self.horizon* self.img_shape[0]*self.img_shape[1]*self.img_shape[2]
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)  # Mean
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)
        self.out_linear1=nn.Linear(self.latent_dim, self.hidden_dim)
        self.out_linear2=nn.Linear(self.hidden_dim,self.out_dim)       
        # self.out_linear3=nn.Linear(self.hidden_dim,self.latent_dim)
        # self.out_linear4=nn.Linear(self.latent_dim,self.out_dim)
    def encode(self, x):
        # x: [B, C, T, H, W]
        
        
        obs_feature=self.obs_encoder(x)
        mu=self.fc_mu(obs_feature)
        logvar=self.fc_logvar(obs_feature)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def key_decode(self, z):
        # z: [B, latent_dim]
        
        batch_size =z.shape[0]
        z = self.out_linear1(z)
        z = F.relu(z)
        z = self.out_linear2(z)
        z = z.view(batch_size, self.horizon, *self.img_shape)
        return z
    # def pre_decode(self, z):
    #     # z: [B, latent_dim]
        
    #     batch_size =z.shape[0]
    #     z = self.out_linear3(z)
    #     z = F.relu(z)
    #     z = self.out_linear4(z)
    #     z = z.view(batch_size, self.horizon, *self.img_shape)
    #     return z
   
    def forward(self, x):
        
        z,_= self.encode(x)
        z=z[:int(z.shape[0]/2),...]# Assuming x is a batch of images with shape [B, C, T, H, W]
        mu=self.fc_mu(z)
        logvar=self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        key_img = self.key_decode(z)
        # pre_img =self.pre_decode(z)
        return key_img,mu, logvar
    def loss(self, x_key,key_img,mu, logvar):
        # Reconstruction loss
        # recon_loss_item = F.mse_loss(pre_img, x_pre, reduction='mean') # Assuming x is the input image sequence
        loss=F.mse_loss(key_img,x_key, reduction='mean') # Assuming x is the input image sequence
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return loss + kl_loss, loss
    def recon_loss(self,x_pre, pre_img):
        recon_loss = F.mse_loss(pre_img, x_pre, reduction='mean')
        return recon_loss