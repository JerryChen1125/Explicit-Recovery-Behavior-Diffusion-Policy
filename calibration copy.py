import cv2
import numpy as np

import pyrealsense2 as rs
from glob import glob

from lerobot.common.datasets.my_utils import create_colored_point_cloud
import open3d as o3d
import os

# Create a context object
ctx = rs.context()

# Get list of all connected devices
devices = ctx.query_devices()

serial_number_dic = {"cam_back": '317622075882',
"cam_arm": '231522072820',
"cam_front": 'f1422097',} 

def compute_rigid_transform(A, B):
    """
    Computes rotation (R) and translation (t) between two point sets A and B.
    A and B must be corresponding points (same size, Nx3).
    """
    assert A.shape == B.shape, "Point sets must have the same shape"
    
    # Center the points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Compute covariance matrix
    H = A_centered.T @ B_centered

    # SVD to get rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_B - R @ centroid_A

    return R, t

def draw(src,tar):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(src)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(tar)

    import copy
    srcDraw = copy.deepcopy(source)
    tarDraw = copy.deepcopy(target)
    srcDraw.paint_uniform_color([1, 0, 0])
    tarDraw.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([srcDraw, tarDraw])
    
def read_points_txt(fname):
    with open(fname,'r') as f:
        lines = f.readlines()
    points = []
    for line in lines:
        point = line.split(' ')
        point = [float(x) for x in point]
        points.append(point)
    return np.array(points)

cam_extrinsics={}
for i,device in enumerate(devices):
    serial_number = device.get_info(rs.camera_info.serial_number)
    print(f"\nDevice with serial number: {serial_number}")
    cam_name = [k for k,v in serial_number_dic.items() if v == serial_number][0]
    # if serial_number[0] != 'f':
    #     continue
    
    # Create a pipeline for this device
    pipeline = rs.pipeline(ctx)
    config = rs.config()
    config.enable_device(serial_number)
    if serial_number[0] == 'f':
        w = 1024
        h = 768
    else:
        w = 640
        h = 480
    config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    
    # Start pipeline
    pipeline.start(config)
    
    try:
        # Get active profile
        profile = pipeline.get_active_profile()
        
        # Get all streams from the profile
        stream = profile.get_stream(rs.stream.depth)
        
        intrinsics = stream.as_video_stream_profile().get_intrinsics()

                
    finally:
        # Stop the pipeline
        pipeline.stop()
        
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
    coeffs = intrinsics.coeffs

    # 已知内参和畸变系数
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array(coeffs, dtype=np.float32)
    
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 7 

    
    # # 3D世界坐标（假设棋盘格在Z=0平面）
    # obj_points = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], dtype=np.float32)
    
    fname = f"captures/camera_{i+1}_{serial_number}/color_1.png"
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    use_full_corners, corners = cv2.findChessboardCorners(gray, (BOARD_WIDTH, BOARD_HEIGHT), None)

    if not use_full_corners:
        txt_name = f"captures_pts/camera_{i+1}.txt"
        corners = read_points_txt(txt_name)
    # 2D图像坐标（通过角点检测得到）
    img_points = corners.squeeze()
    
    if use_full_corners:
        z, y = -np.mgrid[1:11,1:8]
        obj_points = np.vstack(np.stack([np.zeros_like(y),y,z],axis=-1).transpose(1,0,2)).astype(np.float32) * 0.0237
    else:
        x_axis = np.stack([range(7),np.zeros(7),np.zeros(7)],axis=-1)
        z_axis = - np.stack([np.zeros(10),np.zeros(10),range(10)],axis=-1)
        obj_points =  np.vstack([z_axis,x_axis[1:]]).astype(np.float32) * 0.0237
    
    for (x, y) in img_points:
        x, y = int(round(x)), int(round(y))  # Convert to integers
        cv2.circle(img, (x, y), 5, (0, 0, 170), -1)  # Draw green circles

        
    # Display the image
    cv2.imshow('Projected Points', img)
    cv2.waitKey(100000)
    cv2.destroyAllWindows()

    # # 求解外参
    # ret, rvec, tvec,_ = cv2.solvePnPRansac(obj_points, img_points, camera_matrix, dist_coeffs,flags=cv2.SOLVEPNP_ITERATIVE)

    # # 旋转向量转旋转矩阵
    # R, _ = cv2.Rodrigues(rvec)
    
    dimg = cv2.imread(f'captures/camera_{i+1}_{serial_number}/depth_1.png',cv2.IMREAD_UNCHANGED)
    
    xmap = img_points[:,0].astype(int)
    ymap = img_points[:,1].astype(int)
    
    img_points_depth = dimg[ymap,xmap]
    # img_points_depth = img_points_depth[:,0]
    
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    points_z = img_points_depth * depth_scale
    points_x = (xmap - cx) * points_z / fx
    points_y = (ymap - cy) * points_z / fy
    img_cloud_raw = np.stack([points_x, points_y,  points_z], axis=-1)
    img_cloud = img_cloud_raw.reshape([-1, 3]).astype(np.float32)
    img = cv2.projectPoints(img_cloud, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)[0].squeeze()
    
    # img_dist = np.mean(np.sqrt(np.sum((img_cloud[:9]-img_cloud[1:10])**2,axis=-1)))
    # print(img_dist)
    
    # depth_scale = profile.get_device().first_depth_sensor().get_depth_scale() * 0.0237 / img_dist
    # points_z = img_points_depth * depth_scale
    # points_x = (xmap - cx) * points_z / fx
    # points_y = (ymap - cy) * points_z / fy
    # img_cloud_raw = np.stack([points_x, points_y,  points_z], axis=-1)
    # img_cloud = img_cloud_raw.reshape([-1, 3]).astype(np.float32)
    # img = cv2.projectPoints(img_cloud, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)[0].squeeze()

    R, t = compute_rigid_transform(img_cloud, obj_points)
    
    img_cloud_transformed = img_cloud @ R.T + t

    
    
    
    draw(img_cloud_transformed,obj_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.cuda.pybind.utility.Vector3dVector(img_cloud_transformed)
    o3d.io.write_point_cloud('pcd_board.ply',pcd)

    pcd_img = dimg
    extrinsics = {"R":R.tolist(),"t":t.tolist()}
    
    pointclouds = create_colored_point_cloud(pcd_img,pcd_img,intrinsics,extrinsics,depth_scale)
    
    # pointclouds = create_colored_point_cloud(img,img_points_depth,intrinsics,extrinsics,depth_scale)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.cuda.pybind.utility.Vector3dVector(pointclouds)
    pcd_path = f'proj_output/camera_{i+1}_{serial_number}/depth_1.ply'
    os.makedirs(os.path.dirname(pcd_path), exist_ok=True)
    o3d.io.write_point_cloud(pcd_path,pcd)
    extrinsics = {"R":R.tolist(),"t":t.tolist()}
    cam_extrinsics[cam_name] = extrinsics

import json
json_path = '.cache/calibration/gello_default/depth2world.json'


json_string = json.dumps(cam_extrinsics, indent=4)

# Save to file
with open(json_path, "w") as f:
    f.write(json_string)