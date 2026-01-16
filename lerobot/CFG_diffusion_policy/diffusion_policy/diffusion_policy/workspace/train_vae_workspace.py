if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random

import torch.nn.functional as F

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.vae.img_vae import ImageSeqVAE
from diffusion_policy.policy.vae_policy import VAEImagePolicy
# import diffusion_policy.robomimic.utils.obs_utils as ObsUtils
# from diffusion_policy.policy.world_policy import WorldPolicy
OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        self.cfg=cfg
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        # random.seed(seed)
        # ObsUtils.initialize_obs_utils_with_config(cfg)

        # configure model
        
        
        self.model: VAEImagePolicy = hydra.utils.instantiate(cfg.policy)
        # import pdb;
        assert isinstance(self.model, VAEImagePolicy)
        # configure training state
        # print(self.model)
        # exit()
        
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
    def setup_distributed(self):
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
        
        # 初始化分布式训练
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl',rank=0,world_size=2)
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = self.rank  # 在单机多卡中，local_rank通常等于rank
        
        # 设置当前GPU
        torch.cuda.set_device(self.local_rank)
        
        print(f"进程信息: rank={self.rank}, world_size={self.world_size}, local_rank={self.local_rank}")
        print(f"当前使用的GPU: {torch.cuda.current_device()}, GPU名称: {torch.cuda.get_device_name()}")
    def run(self):
        # rank, world_size, gpu = setup_distributed()
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
        # self.setup_distributed()
        # configure dataset
        dataset: BaseImageDataset
        # import pdb;pdb.set_trace()
        # sampler=DistributedSampler(dataset,num_replicas=self.world_size.)
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        # normalizer = dataset.get_normalizer()

        # configure validation dataset
        # val_dataset = dataset.get_validation_dataset()
        # val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
       
       
        # import pdb;pdb.set_trace()

        # device transfer
        device = torch.device(cfg.training.device)
      
        self.model = self.model.to(device)
        optimizer_to(self.optimizer, device)
        self.model.eval()
        # configure lr scheduler
        
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)
               
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    print(train_dataloader.batch_size)
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        # print(batch['action'].shape)
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        # print('-------')
                        # print(batch['obs']['agentview_image'].shape)
                        # # print(batch.keys())
                        # print('-------')
                        # print(batch.keys())
                        # print('-----')

                        # import pdb;pdb.set_trace()
                        output,input,recon_loss= self.model.compute_loss(batch)
                        # print('----')
                        # print(output.shape)
                        # # recon_loss=F.mse_loss(output,input)
                        # # print(raw_loss)
                        # print(recon_loss)
                        print(recon_loss)
                        # print(raw_loss['action_loss'])
                        # loss=raw_loss['recon_loss']+raw_loss['obs_reconstruction_loss']+raw_loss['action_reconstruction_loss']
                        # loss=raw_loss['recon_loss']+raw_loss['kl_loss']+raw_loss['train_scores'].mean()*(-0.01)
                        # print(raw_loss['recon_loss'])
                        # print(raw_loss['train_scores'])
                        loss =recon_loss / cfg.training.gradient_accumulate_every
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                        # print(input.shape)
                        # print(output.shape)
                
                # checkpointss
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    
                    fig, ax = plt.subplots(2, 2, figsize=(8, 4))
                    # import pdb;pdb.set_trace()
                    image1=input[1].permute(1,2,0).detach().cpu().numpy()*255
                    image2=input[0].permute(1,2,0).detach().cpu().numpy()*255
                    print(image1[0][1])
                    ax[0,0].imshow(image1.astype(np.uint8) )
                    ax[0,1].imshow(image2.astype(np.uint8) )
                    ax[1,0].imshow(output[1].permute(1,2,0).detach().cpu().numpy())
                    ax[1,1].imshow(output[0].permute(1,2,0).detach().cpu().numpy())
                    plt.savefig(f'{self.epoch}_yuan_ph.pdf')
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                self.global_step += 1
                self.epoch += 1
                # ========= validate ==========
        joblib.dump(self.model.model.kde, 'kde_model.joblib')
     
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
