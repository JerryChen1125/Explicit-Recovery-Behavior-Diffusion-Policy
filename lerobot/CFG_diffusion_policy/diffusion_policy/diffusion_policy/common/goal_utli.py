import numpy as np
import torch
def find_keyframes_by_state_change(trajectory, threshold,exe_horizon,start_idx):
    # """状态变化幅度大的时刻作为关键帧"""
    keyframes = [start_idx]
 # 初始帧总是关键帧
    length=len(trajectory)-exe_horizon
    
    for i in range(1, length-2*exe_horizon,exe_horizon):
        
        v1=trajectory[i]
        v2=trajectory[i+exe_horizon]
        u1=trajectory[i+2*exe_horizon]
        vector1=np.array(v2)-np.array(v1)
        vector2=np.array(u1)-np.array(v1)
        delta_angle = angle_between_vectors(vector1,vector2)
       
       
        if delta_angle > threshold:
            keyframes.append(i+start_idx)
    
        
    # print(keyframes)
   
    return keyframes
def angle_between_vectors(u, v):
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    cos_theta = dot_product / (norm_u * norm_v)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    return theta_deg

def find_closest_numpy(target, numbers):
    for item in numbers:
        if item > target:
            return temp
        else:
            temp=item
    return numbers[-1]