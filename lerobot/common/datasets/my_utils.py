import numpy as np
import torch
# import pytorch3d.ops as torch3d_ops
from pathlib import Path
from pointnet3 import PointTensor

def create_colored_point_cloud(color, depth, instrinsic, extrinsic, depth_scale, far=3.0, near=0.01, num_points=10000, colored=False, use_grid_sampling=False):
        import time
        begin_time = time.perf_counter()
        
        # assert(depth.shape[0] == color.shape[0] and depth.shape[1] == color.shape[1])
        fx, fy, cx, cy, scale = instrinsic.fx, instrinsic.fy, instrinsic.ppx, instrinsic.ppy, depth_scale
        
    
        # Create meshgrid for pixel coordinates
        xmap = np.arange(depth.shape[1])
        ymap = np.arange(depth.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        end_time = time.perf_counter()
        print("create.2dto3d.meshgrid: " , end_time - begin_time)

        # # Calculate 3D coordinates
        # points_z = depth * scale
        # points_x = (xmap - cx) * points_z / fx
        # points_y = (ymap - cy) * points_z / fy
        # cloud = np.stack([points_x, points_y, points_z], axis=-1)
        # cloud = cloud.reshape([-1, 3])
        # xmap = torch.tensor(xmap, dtype=torch.float, device="cuda")
        # ymap = torch.tensor(xmap, dtype=torch.float, device="cuda")
        # depth = torch.tensor(depth, dtype=torch.float, device="cuda")
        points_z = depth * scale
        mask = np.logical_and(points_z < far, points_z > near)
        points_z = points_z[mask]
        points_x = (xmap[mask] - cx) / fx * points_z
        points_y = (xmap[mask] - cy) / fy * points_z
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        
        
        end_time = time.perf_counter()
        print("create.2dto3d: " , end_time - begin_time)
        
        # # Clip points based on depth
        # mask = (cloud[:, 2] < far) & (cloud[:, 2] > near)
        # cloud = cloud[mask]
        
        if colored:
            # color = color.reshape([-1, 3])
            color = color[mask]

            colored_cloud = np.hstack([cloud, color.astype(np.float32)])
        else:
            colored_cloud = cloud
            
        if use_grid_sampling:
            colored_cloud = grid_sample_pcd(colored_cloud, grid_size=0.005)
        end_time = time.perf_counter()
        print("create.grid_sample: " , end_time - begin_time)
        
        if num_points > colored_cloud.shape[0]:
            num_pad = num_points - colored_cloud.shape[0]
            pad_points = np.zeros((num_pad, 3))
            colored_cloud = np.concatenate([colored_cloud, pad_points], axis=0)
        else: 
            # Randomly sample points
            selected_idx = np.random.choice(colored_cloud.shape[0], num_points, replace=True)
            colored_cloud = colored_cloud[selected_idx]
            
        end_time = time.perf_counter()
        print("create.sample: " , end_time - begin_time)
        
        # shuffle
        np.random.shuffle(colored_cloud)
        
        R = np.array(extrinsic['R'])
        T = np.array(extrinsic['t'])
        
        colored_cloud = colored_cloud @ R.T + T
        return colored_cloud
    
def grid_sample_pcd(point_cloud, grid_size=0.005):
    """
    A simple grid sampling function for point clouds.

    Parameters:
    - point_cloud: A NumPy array of shape (N, 3) or (N, 6), where N is the number of points.
                   The first 3 columns represent the coordinates (x, y, z).
                   The next 3 columns (if present) can represent additional attributes like color or normals.
    - grid_size: Size of the grid for sampling.

    Returns:
    - A NumPy array of sampled points with the same shape as the input but with fewer rows.
    """
    # coords = point_cloud[:, :3]  # Extract coordinates
    # scaled_coords = coords / grid_size
    # grid_coords = np.floor(scaled_coords).astype(int)
    
    # # Create unique grid keys
    # keys = grid_coords[:, 0] + grid_coords[:, 1] * 10000 + grid_coords[:, 2] * 100000000
    
    # # Select unique points based on grid keys
    # _, indices = np.unique(keys, return_index=True)
    
    # # Return sampled points
    # return point_cloud[indices]
    with torch.no_grad():
        coords = torch.as_tensor(point_cloud[:, :3], device="cuda")
        scaled_coords = coords / grid_size
        grid_coords = torch.round(scaled_coords)
        size = grid_coords.max(0)[0] - grid_coords.min(0)[0]
        pt = PointTensor(coords)
        print(size)
        pt.voxelization(grid_size, size.tolist())
        # keys = grid_coords[:, 0] + grid_coords[:, 1] * size[0] + grid_coords[:, 2] * (size[0] * size[1])
        # _, indices = torch.unique(keys)
        # ret = coords[indices]
        ret = pt.grid_center()
    return ret.cpu().numpy()


def point_cloud_sampling(point_cloud:np.ndarray, num_points:int, method:str='fps'):
    """
    support different point cloud sampling methods
    point_cloud: (N, 6), xyz+rgb or (N, 3), xyz
    """
    if num_points == 'all': # use all points
        return point_cloud
    
    if point_cloud.shape[0] <= num_points:
        # cprint(f"warning: point cloud has {point_cloud.shape[0]} points, but we want to sample {num_points} points", 'yellow')
        # pad with zeros
        point_cloud_dim = point_cloud.shape[-1]
        point_cloud = np.concatenate([point_cloud, np.zeros((num_points - point_cloud.shape[0], point_cloud_dim))], axis=0)
        return point_cloud

    if method == 'uniform':
        # uniform sampling
        sampled_indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        point_cloud = point_cloud[sampled_indices]
    elif method == 'fps':
        # fast point cloud sampling using torch3d
        point_cloud = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
        num_points = torch.tensor([num_points]).cuda()
        # remember to only use coord to sample
        # _, sampled_indices = torch3d_ops.sample_farthest_points(points=point_cloud[...,:3], K=num_points)
        point_cloud = point_cloud.squeeze(0).cpu().numpy()
        # point_cloud = point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
    else:
        raise NotImplementedError(f"point cloud sampling method {method} not implemented")

    return point_cloud

def get_point_cloud(rgb_img,depth_img,instrinsic,extrinsic,depth_scale,crop_bound,num_points=512,use_rgb=False):
    import time
    begin_time = time.perf_counter()
    min_bound = crop_bound[1]
    max_bound = crop_bound[0]
    if len(depth_img.shape) > 2:
        point_cloud_array = []
        for depth_item in depth_img:
            point_cloud_unmasked = create_colored_point_cloud(rgb_img,depth_item,instrinsic,extrinsic,depth_scale)
            end_time = time.perf_counter()
            print("create: " , end_time - begin_time)
            
            if not use_rgb:
                point_cloud_unmasked = point_cloud_unmasked[..., :3]
                    
            mask = np.all(point_cloud_unmasked[:, :3] > min_bound, axis=1)
            point_cloud = point_cloud_unmasked[mask]

            mask = np.all(point_cloud[:, :3] < max_bound, axis=1)
            point_cloud = point_cloud[mask]
            end_time = time.perf_counter()
            print("mask: " , end_time - begin_time)

            point_cloud_item = point_cloud_sampling(point_cloud, num_points, 'fps')
            end_time = time.perf_counter()
            print("sample: " , end_time - begin_time)
            point_cloud_array.append(point_cloud_item)
        point_cloud = np.array(point_cloud_array)
    else:
        
        point_cloud_unmasked = create_colored_point_cloud(rgb_img,depth_img,instrinsic,extrinsic,depth_scale)
        end_time = time.perf_counter()
        print("create: " , end_time - begin_time)
        
        if not use_rgb:
            point_cloud_unmasked = point_cloud_unmasked[..., :3]
                
        mask = np.all(point_cloud_unmasked[:, :3] > min_bound, axis=1)
        point_cloud = point_cloud_unmasked[mask]

        mask = np.all(point_cloud[:, :3] < max_bound, axis=1)
        point_cloud = point_cloud[mask]
        end_time = time.perf_counter()
        print("mask: " , end_time - begin_time)

        point_cloud = point_cloud_sampling(point_cloud, num_points, 'fps')
        end_time = time.perf_counter()
        print("sample: " , end_time - begin_time)
    end_time = time.perf_counter()
    print("get_pcd: " , end_time - begin_time)
    
    
    return point_cloud


def read_pcd_frames(pcd_dir, query_ts):
    first_ts = min(query_ts)
    last_ts = max(query_ts)
    
    for ts in query_ts:
        pcd_path = pcd_dir / Path(f"frame{ts:06d}.ply")
        