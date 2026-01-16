from typing import Dict

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import random

from collections import deque
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.prompt.promptembedding import PromptProjector
from diffusion_policy.model.vae.img_vae import ImageSeqVAE
from diffusion_policy.robomimic.models.ae_nets import BasicVAE
class VAEImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
           
            # recon_loss: int,
            horizon: int,
            n_action_steps, 
            n_obs_steps,
            vae: BasicVAE,
            # diffusion model parameters
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        self.n_obs_steps=n_obs_steps
        # # import pdb;pdb.set_trace()
        # # action_shape = shape_meta['action']['shape']
        # # assert len(action_shape) == 1
        # # action_dim = action_shape[0]
        # self.action_dim=action_dim
       
        self.model = vae
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        # self.recon_list=list()
        # self.output_list=list()
    # ========= training  ============
    
    def compute_loss(self, batch):
        # normalize input
        # assert 'valid_mask' not in batch
        # import pdb;pdb.set_trace()
        input=batch
        # import pdb;pdb.set_trace()
        # input_dict={
        #     'agentview_image':input['robot1_eye_in_hand_image'].cuda(),
        #     'robot0_eye_in_hand_image': input['robot0_eye_in_hand_image'].cuda()
        # }
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        input_dict={
            'cam_arm':input['cam_arm'].to('cuda:0'),
            # 'robot0_eye_in_hand_image': input['robot0_eye_in_hand_image'].cuda()
        }
        
        # import pdb;pdb.set_trace()
        # input['image']=input['image']
        # input['agent_pos']=input['agent_pos']
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
    
        reconstructed=self.model(input_dict['cam_arm'][:,0,...])
        # import pdb;pdb.set_trace()
        loss=self.model.loss_function(reconstructed,input_dict['cam_arm'][:,0,...])
        

        return reconstructed,input_dict['cam_arm'][:,0,...],loss
    # def kkd_detect(self,batch):
    #     input=dict_apply(batch, lambda x: x[:,:self.n_obs_steps,...])
    #     input['image']=input['image'].cpu()
    #     input['agent_pos']=input['agent_pos'].cpu()
    #     self.model.KKD_detect(input)