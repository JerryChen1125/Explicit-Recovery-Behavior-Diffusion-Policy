import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Tuple, Union, Dict, Any
import pytorch3d.transforms as pt
import numpy as np
from scipy.spatial.transform import Rotation as R
from ..policy.base_image_policy import BaseImagePolicy



def quat_conjugate(q):

    q = np.asarray(q, dtype=float)
    return q * np.array([1, -1, -1, -1])


def inverse_rotation_axis_angle(q):


    matrix=pt.rotation_6d_to_matrix(torch.tensor(q).cpu().unsqueeze(0)).squeeze(0).numpy()
    return matrix


def matrix_to_rotation_6d(matrix):
    """
    Convert 3x3 rotation matrix to 6D representation.
    Args:
        matrix: [..., 3, 3]
    Returns:
        d6: [..., 6]
    """

    return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6)
def reverse_axis(act):
    reverse_action=-act
    return reverse_action

def reverse_angle(act):
    x = -act[...,0]
    y = -act[...,1]
    z = -act[...,2]
    reverse_action=np.zeros_like(act)

    rot_6d=act[...,3:6]
    rot_6d_inv=np.zeros_like(rot_6d)

    if isinstance(rot_6d, torch.Tensor):


        for i in range(rot_6d.shape[0]):

            rot_matrix = pt.axis_angle_to_matrix(rot_6d.unsqueeze(0)).squeeze(0)
            inv_rot=rot_matrix.T
            rot_6d_inv[i]=pt.matrix_to_axis_angle(inv_rot)


    else:
        for i in range(rot_6d.shape[0]):

            rot_matrix = pt.axis_angle_to_matrix(torch.tensor(rot_6d[i]).unsqueeze(0)).squeeze(0).numpy()

            inv_rot=rot_matrix.T
            rot_6d_inv[i]=pt.matrix_to_axis_angle(torch.from_numpy(inv_rot).unsqueeze(0)).squeeze(0).numpy()
    for i in range(act.shape[0]):

        reverse_action[i][0]=x[i]
        reverse_action[i][1]=y[i]
        reverse_action[i][2]=z[i]
        reverse_action[i][3:6]=rot_6d_inv[i]

    return reverse_action
def reverse(act):
    x = -act[...,0]
    y = -act[...,1]
    z = -act[...,2]
    reverse_action=np.zeros_like(act)

    rot_6d=act[...,3:9]
    rot_6d_inv=np.zeros_like(rot_6d)

    if isinstance(rot_6d, torch.Tensor):


        for i in range(rot_6d.shape[0]):
            for j in range(rot_6d.shape[1]):
                rot_matrix = pt.rotation_6d_to_matrix(rot_6d.unsqueeze(0)).squeeze(0)
                inv_rot=rot_matrix.T
                rot_6d_inv[i][j]=pt.matrix_to_rotation_6d(inv_rot)


    else:

        for i in range(rot_6d.shape[0]):
            for j in range(rot_6d.shape[1]):
                rot_matrix = pt.rotation_6d_to_matrix(torch.tensor(rot_6d[i][j]).unsqueeze(0)).squeeze(0).numpy()

                inv_rot=rot_matrix.T
                rot_6d_inv[i][j]=pt.matrix_to_rotation_6d(torch.from_numpy(inv_rot).unsqueeze(0)).squeeze(0).numpy()
    for i in range(act.shape[0]):
        for j in range(act.shape[1]):
            reverse_action[i][j][0]=x[i][j]
            reverse_action[i][j][1]=y[i][j]
            reverse_action[i][j][2]=z[i][j]
            reverse_action[i][j][3:9]=rot_6d_inv[i][j]
        reverse_action[i]=reverse_action[i][::-1,:]

    return reverse_action
def test_reverse(pos):

    reverse_pos=np.zeros((8,7))

    for j in range(pos.shape[0]):
        pos_matrix=np.eye(4)
        temp_pos=pos[j]

        position=temp_pos[:3]

        orientation=temp_pos[3:6]
        # matrix=inverse_rotation_axis_angle(orientation)
        matrix=pt.axis_angle_to_matrix(torch.tensor(orientation).unsqueeze(0)).squeeze(0).numpy()
        pos_matrix[:3,:3]=matrix
        pos_matrix[:3,3]=position
        pos_matrix_inv=np.linalg.inv(pos_matrix)
        # position=-np.array(position)

        position=pos_matrix_inv[:3,3]
        matrix_inv=pos_matrix_inv[:3,:3]
        rotation_6d=pt.matrix_to_axis_angle(torch.tensor(matrix_inv).unsqueeze(0)).squeeze(0).numpy()
        # rotation_6d=pt.matrix_to_rotation_6d(torch.tensor(matrix_inv).unsqueeze(0)).squeeze(0).numpy()
        pos_array=np.concatenate([position,rotation_6d],axis=0)

        reverse_pos[j][:6]=pos_array
        reverse_pos[j][-1]=temp_pos[-1]
    reverse_pos=reverse_pos[::-1,:]
    reverse_pos=np.array(reverse_pos)

    return reverse_pos
def pushT_reverse(act):
    reverse_pos=act[::-1,:]
    reverse_pos=np.array(reverse_pos)
    return reverse_pos
class error_aware_policy(BaseImagePolicy):
    def __init__(self,policy=None,error_detector=None):
        super().__init__() 
        self.policy = policy
        self.error_detector = error_detector
        self.errors = None
        self.errors_ = None
        self.is_backing=False
        self.backing_end=False
        self.back_time=0
        self.action_shape=None
        self.pre_flag=None
        self.position=[]
        self.backing_step=None
        self.buffer_action = None
        self.step=-1
        self.prompt_buffer = None
        self.moving=False
    def add_error(self, error):
        # print(error)
        # error = mse.mean(dim=[1,2,3]).detach().cpu().numpy()
        for i in range(error.shape[0]):
            self.errors[i].append([error[i]])
        for i in range(error.shape[0]):
            self.errors_[i].append([error[i]])
    def add_qpos(self,obs):
        pos=np.array(obs['robot0_eef_pos'][:,1,:].cpu().detach().numpy())
        quat=np.array(obs['robot0_eef_quat'][:,1,:].cpu().detach().numpy())
        pos=np.concatenate([pos,quat],axis=-1)
        self.position.append(pos)
    def get_error_statistics(self):
        if not self.errors:
            return None
        mean_array=np.zeros(len(self.errors))
        std_array=np.zeros(len(self.errors))
        errors= [[] for _ in range(len(self.errors))]
        for i in range(len(self.errors)):
            errors[i].append(np.concatenate(self.errors[i],axis=0))
        for i in range(len(errors)):
            mean_array[i]=np.mean(errors[i],axis=1)
            std_array[i]=np.mean(errors[i],axis=1)

        print(self.errors)
        print(f"mean_array{mean_array}")
        print(f"std_array{std_array}")
        return {
            "mean":mean_array,
            "std":std_array
        }
    def reset_errors(self,i):
        self.errors[i]=[]
    def draw_back(self,pre_flag,flag,observation):
        if not self.buffer_action:
            return None  # No action to draw back to
        action=np.zeros(self.action_shape)
        for i in range(action.shape[0]):
            action[i]=pushT_reverse(self.buffer_action[i][-1].cpu().detach().numpy())


        action_pred=self.policy.predict_action(observation)
        for i in range(1):
            if pre_flag:
                if flag:
                    # import pdb;pdb.set_trace()
                    # print('still_backing')
                    # action[i]=action[i]
                    # self.backing_step[i]=self.backing_step[i]+1
                    observation_per=dict()
                    for key in observation.keys():
                        observation_per[key]=observation[key][i].unsqueeze(0)
                    action_negative= self.policy.predict_action(observation_per,negative_prompt=self.prompt_buffer[i][-1].unsqueeze(0))
                    action[i]=action_negative['action'].squeeze(0).cpu().detach().numpy()
                    self.reset_errors(i)
                else:
                    print('negative')
                    observation_per=dict()
                    for key in observation.keys():
                        observation_per[key]=observation[key][i].unsqueeze(0)
                    action_negative= self.policy.predict_action(observation_per,negative_prompt=self.prompt_buffer[i][-1].unsqueeze(0))
                    action[i]=action_negative['action'].squeeze(0).cpu().detach().numpy()
                    self.reset_errors(i)
            else:
                if flag:
                    action[i]=action[i]

                    print('backing')
                    print('----------------------------------------------')
                    self.backing_step[i]=self.backing_step[i]+1
                else: 
                    print('forward')
                    observation_per=dict()
                    # print(action_pred['action'])
                    self.buffer_action[i].append(action_pred['action'][i])
                    self.prompt_buffer[i].append(action_pred['action_pred'][i])
                    action[i]=action_pred['action'].cpu().detach().numpy()[i]
        action=np.array(action.tolist()).astype(float)
        action=action.astype(float)
        return action

    def detect_error(self, observation):
        if self.error_detector is None:
            print('returning')
            return False
        # image= torch.tensor(observation['image'][:,0,...]).cuda().float()
        # recon_image = self.error_detector(image)
        # mse = torch.nn.functional.mse_loss(recon_image, image, reduction='none')
        # # import pdb;pdb.set_trace()
        # error = mse.mean(dim=[1,2,3]).detach().cpu().numpy()
        # print(error.shape)          
                    # wandb_run.log()
        # import pdb;pdb.set_trace()
        print('detecting')
        _,_,loss=self.error_detector.compute_loss(observation)
        # loss=0
        error=torch.tensor(loss).unsqueeze(0)
       
        # error = mse.mean()
        error=error.detach().cpu().numpy()
        # print('-----')
        # print(error.shape)
        # print(error.shape)
        # print('------')
        self.add_error(error)
        if len(self.errors)==0:
            return [False]
        else:
            self.threshold = self.get_error_statistics()['mean'] +0.3* self.get_error_statistics()['std']
            print(f'error{error}')
            print(f'threshold{self.threshold}')
            self.threshold=np.array(self.threshold)
            return error>self.threshold
    def predict_action(self, observation):
        print('moving')
        batch_size=observation['cam_arm'].shape[0] 
        if self.moving:

            # for i in range(len(done)):
            #     if done[i]:
            #         import pdb;pdb.set_trace()
            #         self.errors[i]=[]
            flag=self.detect_error(observation)
            action=self.draw_back(self.pre_flag,flag,observation)
            self.pre_flag=flag
            # print(self.backing_step)
            # print(np.array(self.backing_step).sum())
            return action
        else:
            # print(self.moving)
            # print(done)
            # for i in range(len(done)):
            #     if done[i]:
            #         self.errors[i]=[]
            self.moving=True
            self.errors=[[] for _ in range(batch_size)]
            self.errors_=[[] for _ in range(batch_size)]
            self.buffer_action=[[] for _ in range(batch_size)]
            self.prompt_buffer=[[] for _ in range(batch_size)]
            self.backing_step=np.ones(batch_size, dtype=int)  
            # print(self.backing_step)
            # self.is_backing=torch.zeros(observation.shape[0])>torch.ones(observation.shape[0])
            self.pre_flag=self.detect_error(observation)
            action=self.policy.predict_action(observation)
            for i in range(batch_size):
                self.buffer_action[i].append(action['action'][i].squeeze(0))
                self.prompt_buffer[i].append(action['action_pred'][i])
            # self.add_qpos(observation)
            self.action_shape=action['action'].shape
            return action['action'].cpu().detach().numpy()