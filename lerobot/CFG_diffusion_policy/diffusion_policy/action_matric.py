import matplotlib.pyplot as plt
import numpy as np
import pickle
def angle_between_vectors(v1, v2):
    """Returns the angle in radians between vectors v1 and v2"""
    v1_u = v1 / np.linalg.norm(v1)  # Unit vector
    v2_u = v2 / np.linalg.norm(v2)  # Unit vector
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

with open('action_step_base.pkl', 'rb') as file:
    load_data=np.array(pickle.load(file))
    # print(load_data.pop())

with open('action_step_exp.pkl','rb') as file:
    load_data1=np.array(pickle.load(file))
print(load_data.shape)
print(load_data1.shape)
for i in range(load_data.shape[0]):
    if i ==0:
        data=load_data[i]
    else:
        temp_data=load_data[i]
        data=np.concatenate([data,temp_data],axis=-2)
for i in range(load_data1.shape[0]):
    if i ==0:
        data1=load_data1[i]
    else:
        temp_data=load_data1[i]
        data1=np.concatenate([data1,temp_data],axis=-2)  
print(data.shape)
print(data1.shape)
deg=0
for i in range(data.shape[0]):
    # print(data[i])
    

    angle_rad = angle_between_vectors(data[i][0], data[i][0])
        # print(angle_rad)
    angle_deg = np.degrees(angle_rad)
        
    deg=deg+angle_deg
print(deg/data.shape[0])