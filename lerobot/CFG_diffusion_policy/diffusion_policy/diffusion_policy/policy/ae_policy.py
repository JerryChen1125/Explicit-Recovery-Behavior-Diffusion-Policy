from typing import Dict

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import random
import numpy as np
from collections import deque
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.prompt.promptembedding import PromptProjector
from diffusion_policy.ae.model.ae_model import Network
class AEImagePolicy(BaseImagePolicy):
    def __init__(self, 
            ae: Network,
            # parameters passed to step
            ):
        super().__init__()

        
        self.model = ae
        
        
    # ========= training  ============
    def loss_function(self, recon_x, x):
        # loss = F.mse_loss(x, recon_x, reduction='mean') 
        BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')
        return BCE
    def compute_loss(self,batch):
        # batch=batch.to(self.device)
        # print(batch['obs']['image'].shape)
        
        # batch['image']=np.array(batch['obs']['image'].float().cpu()/255.0)
        # image = np.moveaxis(batch['obs']['image'],1,2)
        # print(batch['obs']['image'].shape)
        image = batch['image'][14:16].cpu().float()
        # print(image)
        # print(image.shape)
        # batch['image']=np.array(batch['image'].float().cpu()/255.0)
        # image = np.moveaxis(batch['image'],4,2)
        # print(f'image{image.shape}')
        # image=np.moveaxis(batch['image'],1,2)
        # image=torch.from_numpy(image)
        # print(f'image{image.shape}')
        # print(batch['obs']['image'].shape)
        recon_batch=self.model(image)
        # print(recon_batch)
        # print(recon_batch.shape)
        # recon_batch=self.model(image)
        loss=self.loss_function(recon_batch,image)

        test_batch=torch.rand_like(image)
        test_recon=self.model(test_batch)
        test_loss=self.loss_function(test_batch,test_recon)
       
        print(f'test_loss:{test_loss}')
        print(f'loss:{loss}')
        return loss
    # def compute_loss(self, batch):
    #     # normalize input
    #     # assert 'valid_mask' not in batch
    #     input=dict_apply(batch['obs'], lambda x: x[:,:self.n_obs_steps,...])
        
    #     return_dict=self.model(input,batch['pre'], batch['goal'],conditions=batch['pre'])

