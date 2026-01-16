import cv2
import numpy as np

# 已知内参和畸变系数
camera_matrix = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)

# 3D世界坐标（假设棋盘格在Z=0平面）
obj_points = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], dtype=np.float32)

# 2D图像坐标（通过角点检测得到）
img_points = np.array([[0,0], [0,1], [1,1], [1,0]], dtype=np.float32)

# 求解外参
ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix,dist_coeffs,flags=cv2.SOLVEPNP_EPNP)

# 旋转向量转旋转矩阵
R, _ = cv2.Rodrigues(rvec)
print("Rotation Matrix:\n", R)
print("Translation Vector:\n", tvec)