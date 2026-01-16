import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.animation import FuncAnimation
def angle_between_vectors(v1, v2):
    """Returns the angle in radians between vectors v1 and v2"""
    v1_u = v1 / np.linalg.norm(v1)  # Unit vector
    v2_u = v2 / np.linalg.norm(v2)  # Unit vector
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
with open('action_step_real.pkl', 'rb') as file:
    load_data=pickle.load(file)
    
# load_data_backup=list()
# for i in range(load_data.shape[0]):
#     for j in range(load_data[1].shape[0]):
#         load_data[i][j]=load_data[i][1]
#         print(load_data[i][j])
# load_data=list(load_data)
# with open('pre_action.pkl','wb') as file:
#     pickle.dump(load_data,file)
with open('action_step_negative_prompt.pkl','rb') as file:
    # print(load_data.pop())
    load_data1=pickle.load(file)
    # print(load_data)

# print(load_data1.shape)
for i in range(len(load_data)):
    if i ==0:
        data=load_data[i].cpu()
    else:
        temp_data=load_data[i].cpu()
        data=np.concatenate([data,temp_data],axis=-2)
for i in range(len(load_data)):
    if i ==0:
        data1=load_data1[i].cpu()
    else:
        temp_data=load_data1[i].cpu()
        data1=np.concatenate([data1,temp_data],axis=-2)
print(data.shape)
print(data1.shape)
fig, ax = plt.subplots()

point, = ax.plot([], [], 'ro')  # 空线（红色实线）
point1, = ax.plot([], [], 'bo')  # 空线（红色实线）
ax.set_xlim(-500, 500)
ax.set_ylim(-500, 550)
tra=data[3]
tra1=data1[3]

def update(frame):
    # print(frame)
    x = tra[:frame,0]
    y = tra[:frame,1]
    x1 = tra1[:frame+1,0]
    
    y1 = tra1[:frame+1,1]  # 动态相位
    point.set_data(x, y)  
    point1.set_data(x1, y1)  
       # 更新线数据
    return point1,point
ani = FuncAnimation(
    fig, update, frames=304, 
    interval=100, blit=True
)
# ani.save('sine_wave.gif', writer='pillow', fps=1) 
plt.title('no_prompt')
plt.show()


# plt.plot(x, y, label='sin(x)')
# plt.xlabel('X 轴')
# plt.ylabel('Y 轴')
# plt.title('二维轨迹可视化')
# plt.legend()
# plt.grid(True)
# plt.show()