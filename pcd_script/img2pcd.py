import os
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import json
import time
from multiprocessing import Pool, cpu_count
import argparse

class PointCloudFusion:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.meta_path = Path("pcd_meta")
        
        inst_file = self.meta_path / f"inst.json"
        calib_file = self.meta_path / f"calib.json"
        
        # 加载相机参数
        self.camera_params = self.load_camera_parameters()
        
        """加载三个相机的内外参矩阵"""
        params = {}
        for cam_name in ["cam_left", "cam_right", "cam_arm"]:
            try:
                # 假设参数存储在JSON文件中
                with open(inst_file, 'r') as f:
                    params[cam_name] = json.load(f)
                    
            except Exception as e:
                print(f"Error loading camera {cam_id} parameters: {e}")
        
        return params
    
    def load_frame_parameters(self, frame_idx):
        """加载第三相机每一帧的外参（眼在手上）"""
        try:
            param_file = self.meta_path / "camera3_frames" / f"frame_{frame_idx:06d}.npy"
            if param_file.exists():
                return np.load(param_file)
        except:
            pass
        return None
    
    def depth_to_pointcloud(self, rgb_image, depth_image, intrinsic, extrinsic=None):
        """将RGB-D图像转换为点云"""
        height, width = depth_image.shape
        
        # 创建像素坐标网格
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)
        
        # 将深度图转换为米（假设深度图存储的是毫米）
        depth_meters = depth_image.astype(np.float32) / 1000.0
        
        # 有效深度掩码
        valid_mask = (depth_meters > 0) & (depth_meters < 10.0)  # 假设深度在10米内有效
        
        # 反投影到3D空间
        z = depth_meters[valid_mask]
        x = (u[valid_mask] - intrinsic[0, 2]) * z / intrinsic[0, 0]
        y = (v[valid_mask] - intrinsic[1, 2]) * z / intrinsic[1, 1]
        
        # 组合点云
        points = np.stack([x, y, z], axis=-1)
        
        # 获取颜色
        colors = rgb_image[valid_mask] / 255.0
        
        # 应用外参变换（如果提供）
        if extrinsic is not None:
            # 转换为齐次坐标
            points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
            points_transformed = (extrinsic @ points_homo.T).T
            points = points_transformed[:, :3]
        
        return points, colors
    
    def process_single_camera(self, camera_id, frame_files):
        """处理单个相机的点云生成"""
        all_pointclouds = []
        
        for rgb_file in frame_files:
            # 提取帧号
            frame_num = rgb_file.stem
            
            # 构建对应的深度文件路径
            depth_file = self.depth_paths[camera_id-1] / f"{frame_num}.png"
            
            if not depth_file.exists():
                continue
            
            try:
                # 读取RGB和深度图像
                rgb_image = cv2.imread(str(rgb_file))
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                depth_image = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
                
                if rgb_image is None or depth_image is None:
                    continue
                
                # 获取相机参数
                intrinsic = self.camera_params[camera_id]['intrinsic']
                extrinsic = self.camera_params[camera_id]['extrinsic']
                
                # 如果是第三相机（眼在手上），加载帧特定的外参
                if camera_id == 3:
                    frame_extrinsic = self.load_frame_parameters(int(frame_num))
                    if frame_extrinsic is not None:
                        extrinsic = frame_extrinsic
                
                # 生成点云
                points, colors = self.depth_to_pointcloud(rgb_image, depth_image, intrinsic, extrinsic)
                
                if len(points) > 0:
                    all_pointclouds.append((points, colors))
                    
            except Exception as e:
                print(f"Error processing camera {camera_id}, frame {frame_num}: {e}")
                continue
        
        return all_pointclouds
    
    def merge_pointclouds(self, pointclouds_list):
        """合并所有点云"""
        all_points = []
        all_colors = []
        
        for camera_pointclouds in pointclouds_list:
            for points, colors in camera_pointclouds:
                all_points.append(points)
                all_colors.append(colors)
        
        if not all_points:
            return None, None
            
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)
        
        return merged_points, merged_colors
    
    def save_pointcloud(self, points, colors, filename):
        """保存点云为PLY文件"""
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 保存点云
        output_file = self.output_path / filename
        o3d.io.write_point_cloud(str(output_file), pcd)
        print(f"Saved point cloud: {output_file}")
    
    def process_single_frame_parallel(self, args):
        """并行处理单帧的包装函数"""
        return self.process_single_frame(*args)
    
    def process_single_frame(self, frame_files):
        """处理单帧（三个相机）"""
        frame_num = Path(frame_files[0]).stem
        
        print(f"Processing frame {frame_num}...")
        
        all_camera_pointclouds = []
        
        for camera_id in range(3):
            rgb_file = frame_files[camera_id]
            depth_file = self.depth_paths[camera_id] / f"{frame_num}.png"
            
            if not depth_file.exists():
                continue
                
            try:
                # 读取图像
                rgb_image = cv2.imread(str(rgb_file))
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                depth_image = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
                
                if rgb_image is None or depth_image is None:
                    continue
                
                # 获取相机参数
                intrinsic = self.camera_params[camera_id+1]['intrinsic']
                extrinsic = self.camera_params[camera_id+1]['extrinsic']
                
                # 第三相机特殊处理
                if camera_id == 2:
                    frame_extrinsic = self.load_frame_parameters(int(frame_num))
                    if frame_extrinsic is not None:
                        extrinsic = frame_extrinsic
                
                # 生成点云
                points, colors = self.depth_to_pointcloud(rgb_image, depth_image, intrinsic, extrinsic)
                
                if len(points) > 0:
                    all_camera_pointclouds.append((points, colors))
                    
            except Exception as e:
                print(f"Error processing camera {camera_id+1}, frame {frame_num}: {e}")
                continue
        
        # 合并点云
        if all_camera_pointclouds:
            points_list = [pc[0] for pc in all_camera_pointclouds]
            colors_list = [pc[1] for pc in all_camera_pointclouds]
            
            merged_points = np.vstack(points_list)
            merged_colors = np.vstack(colors_list)
            
            # 保存合并后的点云
            self.save_pointcloud(merged_points, merged_colors, f"frame_{frame_num}.ply")
            return True
        
        return False
    
    def run_sequential(self):
        """顺序处理所有帧"""
        # 获取所有RGB文件（以第一个相机为准）
        rgb_files = sorted(list(self.rgb_paths[0].glob("*.png")) + 
                          list(self.rgb_paths[0].glob("*.jpg")))
        
        total_frames = len(rgb_files)
        processed_count = 0
        
        for rgb_file in rgb_files:
            frame_num = rgb_file.stem
            
            # 检查所有相机是否有对应的文件
            frame_files = []
            valid_frame = True
            for i in range(3):
                corresponding_file = self.rgb_paths[i] / f"{frame_num}.png"
                if not corresponding_file.exists():
                    corresponding_file = self.rgb_paths[i] / f"{frame_num}.jpg"
                    if not corresponding_file.exists():
                        valid_frame = False
                        break
                frame_files.append(corresponding_file)
            
            if valid_frame:
                success = self.process_single_frame(frame_files)
                if success:
                    processed_count += 1
            
            print(f"Progress: {processed_count}/{total_frames}")
    
    def run_parallel(self, num_processes=None):
        """并行处理所有帧"""
        if num_processes is None:
            num_processes = min(cpu_count(), 8)  # 限制最大进程数
        
        # 获取所有有效的帧文件组
        frame_groups = []
        rgb_files = sorted(list(self.rgb_paths[0].glob("*.png")) + 
                          list(self.rgb_paths[0].glob("*.jpg")))
        
        for rgb_file in rgb_files:
            frame_num = rgb_file.stem
            frame_files = []
            valid_frame = True
            
            for i in range(3):
                corresponding_file = self.rgb_paths[i] / f"{frame_num}.png"
                if not corresponding_file.exists():
                    corresponding_file = self.rgb_paths[i] / f"{frame_num}.jpg"
                    if not corresponding_file.exists():
                        valid_frame = False
                        break
                frame_files.append(corresponding_file)
            
            if valid_frame:
                frame_groups.append(frame_files)
        
        print(f"Starting parallel processing with {num_processes} processes...")
        print(f"Total frames to process: {len(frame_groups)}")
        
        # 使用进程池并行处理
        with Pool(processes=num_processes) as pool:
            results = []
            for i, frame_files in enumerate(frame_groups):
                result = pool.apply_async(self.process_single_frame, (frame_files,))
                results.append(result)
                
                # 限制同时提交的任务数量，避免内存溢出
                if len(results) >= num_processes * 2:
                    for result in results:
                        result.get()
                    results = []
            
            # 等待剩余任务完成
            for result in results:
                result.get()

def main():
    parser = argparse.ArgumentParser(description='RGB-D Point Cloud Fusion')
    parser.add_argument('--data_path', type=str, default="/home/sealab/lerobot_outputs/test_task", help='Path to the data directory')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes for parallel processing')
    
    args = parser.parse_args()
    
    # 初始化点云融合器
    fusion = PointCloudFusion(args.data_path)
    
    # 检查相机参数是否加载成功
    if not fusion.camera_params:
        print("Error: Could not load camera parameters!")
        return
    
    print("Camera parameters loaded successfully:")
    for cam_id, params in fusion.camera_params.items():
        print(f"Camera {cam_id}: Intrinsic shape {params['intrinsic'].shape}, Extrinsic shape {params['extrinsic'].shape}")
    
    start_time = time.time()
    
    # 运行处理
    if args.parallel:
        fusion.run_parallel(args.processes)
    else:
        fusion.run_sequential()
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()