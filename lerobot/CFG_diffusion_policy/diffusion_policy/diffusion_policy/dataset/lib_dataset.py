import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import random
import pathlib
import IPython
import copy
from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
e = IPython.embed
def libero_buffer(dataset_dir,horizon,pad_before=0,pad_after=0):
    file_path=pathlib.Path(dataset_dir)
    indices=[]
    hdf5=[ hdf5_file for hdf5_file in file_path.glob("*.hdf5")]
    for hdf5_file in hdf5:
        with h5py.File(hdf5_file, 'r') as root: 
            replay_buffer = root['data']
            demo_keys=root['data'].keys()
            for demo in demo_keys:
                episode=root['data'][demo]
                episode_len=episode['actions'].shape[0]
                min_start = -pad_before
                max_start = episode_len - horizon + pad_after
                for idx in range(min_start, max_start+1):
                    buffer_start_idx = max(idx, 0)
                    buffer_end_idx = min(idx+horizon, episode_len)
                    start_offset = buffer_start_idx -idx
                    end_offset = (idx+horizon) - buffer_end_idx
                    sample_start_idx = 0 + start_offset
                    sample_end_idx = horizon - end_offset
                # if debug:
                #     assert(start_offset >= 0)
                #     assert(end_offset >= 0)
                #     assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
                    indices.append([
                    hdf5_file,demo,buffer_start_idx, buffer_end_idx, 
                    sample_start_idx, sample_end_idx])
                    
    return indices
class EpisodicDataset(BaseImageDataset):
    def __init__(self, dataset_dir,horizon,n_action_steps,n_obs_steps,pad_before=0,pad_after=0):
        super(EpisodicDataset).__init__()
        
        self.sequence_length = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.dataset_dir = dataset_dir
        self.is_sim = None
        image,state= self.load_data()
        self.replay_buffer = {
            'image': image,
            'action': state,
        }
        self.indices=libero_buffer(dataset_dir,horizon,pad_before=pad_before,pad_after=pad_after)
        print(len(self.indices))
        # self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode
        
        episode_id, demo_key, buffer_start_idx,buffer_end_idx,start_idx, end_idx = self.indices[index]
        result=dict()
        
               
        # dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        for key in ['obs/agentview_rgb','actions']:
            with h5py.File(episode_id, 'r') as root:
                # self.demo_key=demo_key
                # print(type(demo_key))
                # print(key)
                
                input_arr = root['data'][demo_key]
                input_arr=input_arr[key][()]
            # # for data_key in  ['obs/agentview_image', 'obs/ee_states']:
            # #     if data_key not in episode:
            # #         raise KeyError(f"Key {data_key} not found in episode {episode_id}!")
            # #     data_key=episode[data_key][start_idx:end_idx] # (T, H, W, C)
            # image=episode['obs/agentview_image'][start_idx:end_idx] # (T, H, W, C)
            # state=episode['obs/ee_states'][start_idx:end_idx] # (T,
            #      # (T, state_dim)
            # # qpos=episode['obs/qpos'][start_idx:end_idx] # (T,
           
                # performance optimization, only load used obs steps
            sample = input_arr[buffer_start_idx:buffer_end_idx]
            if (start_idx > 0) or (end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if start_idx > 0:
                    data[:start_idx] = sample[0]
                if end_idx < self.sequence_length:
                    data[end_idx:] = sample[-1]
                data[start_idx:end_idx] = sample
            else:
                data=sample
            result[key] = data


        # construct observations
        image_data = torch.from_numpy(result['obs/agentview_rgb'])
        qpos_data = torch.from_numpy(result['actions']).float()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = image_data / 255.0
        data = {
            'obs': {
                'image': image_data, # T, 3, 96, 96
                # 'action': qpos_data, # T, 2
            },
            'action': qpos_data # T, 2
        }
        return data
        # normalize image and change dtype to float
        


    # def get_validation_dataset(self):
    #     val_set = copy.copy(self)
    #     val_set.sampler = SequenceSampler(
    #         replay_buffer=self.replay_buffer, 
    #         sequence_length=self.horizon,
    #         n_obs_steps=self.n_obs_steps,
    #         n_action_steps=self.n_action_steps,
    #         pad_before=self.pad_before, 
    #         pad_after=self.pad_after,
    #         episode_mask=~self.train_mask
    #         )
    #     val_set.train_mask = ~self.train_mask
    #     return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data={
            'action': self.replay_buffer['action']}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer
    def load_data(self):
        file_path=pathlib.Path(self.dataset_dir)
        indices=[]
        hdf5=[ hdf5_file for hdf5_file in file_path.glob("*.hdf5")]
        img=list()
        sta=list()
        for hdf5_file in hdf5:
            with h5py.File(hdf5_file, 'r') as root: 
                # replay_buffer = root['data']
                demo_keys=root['data'].keys()
                # print(demo_keys)
                
                for dmeo in demo_keys:
                    image=root['data'][dmeo]['obs/agentview_rgb'][()]
                    # print(image)
                    state=root['data'][dmeo]['actions'][()]
                    # img.append(image)
                    sta.append(state)
        image=torch.from_numpy(np.concatenate(image,axis=0))
        state=torch.from_numpy(np.concatenate(sta,axis=0)).float()
        return image, state
    #     action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    #     action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    #     action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # # normalize qpos data
    #     qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    #     qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    #     qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    #     stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
    #          "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
    #          "example_qpos": qpos}

                    # image_data = torch.from_numpy(image)
                    # state_data = torch.from_numpy(state).float()
                    
def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)