from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.libero_read import HDF5Buffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
import random
import pathlib
class LiberalDataset(BaseImageDataset):
    def __init__(self, 
            file_path, 
            horizon=1,
            n_obs_steps=1,
            n_action_steps=2,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        file_path=pathlib.Path(file_path)
        hdf5=[ hdf5_file for hdf5_file in file_path.glob("*.hdf5")]
        file=random.choice(hdf5)
        self.replay_buffer = HDF5Buffer(
            file)
        # val_mask = get_val_mask(
        #     n_episodes=self.replay_buffer.n_episodes, 
        #     val_ratio=val_ratio,
        #     seed=seed)
        # train_mask = ~val_mask
        # train_mask = downsample_mask(
        #     mask=train_mask, 
        #     max_n=max_train_episodes, 
        #     seed=seed)
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            )
       
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_action_steps=n_action_steps

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set


    def get_normalizer(self, mode='limits', **kwargs):
        
        data = self._sample_to_data(self.replay_buffer)['obs']
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        
        agent_pos = sample['obs/ee_states'][()].astype(np.float32) # (agent_posx2, block_posex3)
        
        # goal_pos = sample['goal_state'][:,:2].astype(np.float32) # (goal_posx2, block_posex3)
        # pre_pos = sample['pre_state'][:,:2].astype(np.float32)
        # keypoint = sample[self.obs_key]
        # state = sample[self.state_key]
        # agent_pos = agent[:,:2]
        # obs = np.concatenate([
        #     keypoint.reshape(keypoint.shape[0], -1), 
        #     agent_pos], axis=-1)
        image=sample['obs/agentview_rgb'].astype(np.float32)
       
        image = np.moveaxis(image,-1,1)/255.0
        # data=sample
        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            }# T, D_o
            # 'action': sample[actions], # T, D_a
        }
       
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
