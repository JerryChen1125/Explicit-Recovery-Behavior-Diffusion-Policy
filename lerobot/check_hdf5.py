import h5py
import numpy as np


with h5py.File('/home/sealab/jr_code/lerobot/lerobot/task1.hdf5', 'r') as f:
    data = f['data']
    obs=data['obs']
    cam_arm=obs['cam_arm'][()]
    cam_right=obs['cam_right'][()]
    cam_left=obs['cam_left'][()]
    print("Camera arm shape:", cam_arm.shape)
    print("Camera right shape:", cam_right.shape)       