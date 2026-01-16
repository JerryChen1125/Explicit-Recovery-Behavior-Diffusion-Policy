import pyrealsense2 as rs
import numpy as np
import cv2
import os

cam_name2sn = {"cam_arm": '317622075882',
"cam_right": '215222072518',
"cam_left": '231522072820'}

def align_images_using_intrinsics(rgb_path, depth_path, output_dir):
    """
    使用相机内参将RGB图像对齐到深度图像
    
    参数:
        rgb_path: RGB图像路径
        depth_path: 深度图像路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    cam_name = [name for name in cam_name2sn.keys() if name in rgb_path][0]
    cam_SN = cam_name2sn[cam_name]
    
    # 读取图像
    rgb_image = cv2.imread(rgb_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    
    if rgb_image is None or depth_image is None:
        print("无法读取图像文件")
        return
    
    # 配置相机管道
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 启用彩色和深度流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_device(cam_SN)
    
    # 启动管道
    pipeline_profile = pipeline.start(config)
    
    # 获取相机内参
    color_profile = pipeline_profile.get_stream(rs.stream.color)
    depth_profile = pipeline_profile.get_stream(rs.stream.depth)
    
    rgb_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        
    # RGB到深度的外参变换（深度到彩色的逆变换）
    depth_to_color_extrinsics = depth_profile.get_extrinsics_to(color_profile)
    color_to_depth_extrinsics = color_profile.get_extrinsics_to(depth_profile)
    
    # 将RGB对齐到深度
    aligned_rgb = align_rgb_to_depth(rgb_image, depth_image, rgb_intrinsics, 
                                   depth_intrinsics, color_to_depth_extrinsics)
    
    # 保存对齐后的图像
    cv2.imwrite(os.path.join(output_dir, "original_rgb.png"), rgb_image)
    cv2.imwrite(os.path.join(output_dir, "original_depth.png"), depth_image)
    cv2.imwrite(os.path.join(output_dir, "aligned_rgb_to_depth.png"), aligned_rgb)
    cv2.imwrite(os.path.join(output_dir, "original_depth_vis.png"), 
                cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET))
    
    # 创建可视化图像
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
    )
    
    # 水平拼接显示
    combined = np.hstack((aligned_rgb, depth_colormap))
    cv2.imwrite(os.path.join(output_dir, "combined_visualization.png"), combined)
    
    # 停止管道
    pipeline.stop()
    
    print(f"对齐完成！结果保存在: {output_dir}")

def align_rgb_to_depth(rgb_frame, depth_frame, rgb_intrinsics, depth_intrinsics, color_to_depth_extrinsics):
    """
    将RGB图像对齐到深度图像坐标系
    """
    rgb_image = np.array(rgb_frame)
    depth_image = np.array(depth_frame)
    
    # 创建对齐后的RGB图像（与深度图像相同尺寸）
    aligned_rgb = np.zeros((depth_intrinsics.height, depth_intrinsics.width, 3), dtype=rgb_image.dtype)
    
    # 创建权重图以避免空洞
    weight_map = np.zeros((depth_intrinsics.height, depth_intrinsics.width), dtype=np.int32)
    
    for v_r in range(rgb_intrinsics.height):
        for u_r in range(rgb_intrinsics.width):
            # 获取RGB像素颜色
            color = rgb_image[v_r, u_r]
            
            # 将RGB像素坐标转换为3D点
            # 由于RGB图像没有深度信息，我们需要使用深度图像中的对应深度
            # 这里我们使用反向映射：先找到深度图中对应的点
            
            # 对于RGB到深度的对齐，更有效的方法是遍历深度图并查找对应的RGB像素
            pass
    
    # 更有效的方法：遍历深度图像的每个像素，找到对应的RGB像素
    for v_d in range(depth_intrinsics.height):
        for u_d in range(depth_intrinsics.width):
            depth_value = depth_image[v_d, u_d]
            
            if depth_value == 0:
                continue
                
            # 将深度像素坐标转换为3D点
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u_d, v_d], depth_value)
            
            # 变换到RGB相机坐标系
            rgb_point = rs.rs2_transform_point_to_point(color_to_depth_extrinsics, depth_point)
            
            # 将3D点投影到RGB图像平面
            rgb_pixel = rs.rs2_project_point_to_pixel(rgb_intrinsics, rgb_point)
            
            u_r = int(round(rgb_pixel[0]))
            v_r = int(round(rgb_pixel[1]))
            
            # 检查是否在RGB图像范围内
            if 0 <= u_r < rgb_intrinsics.width and 0 <= v_r < rgb_intrinsics.height:
                aligned_rgb[v_d, u_d] = rgb_image[v_r, u_r]
    
    return aligned_rgb

def align_rgb_to_depth_advanced(rgb_frame, depth_frame, rgb_intrinsics, depth_intrinsics, color_to_depth_extrinsics):
    """
    高级版本的RGB到深度对齐，使用双线性插值减少空洞
    """
    rgb_image = np.array(rgb_frame)
    depth_image = np.array(depth_frame)
    
    aligned_rgb = np.zeros((depth_intrinsics.height, depth_intrinsics.width, 3), dtype=rgb_image.dtype)
    
    for v_d in range(depth_intrinsics.height):
        for u_d in range(depth_intrinsics.width):
            depth_value = depth_image[v_d, u_d]
            
            if depth_value == 0:
                continue
                
            # 将深度像素坐标转换为3D点
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u_d, v_d], depth_value)
            
            # 变换到RGB相机坐标系
            rgb_point = rs.rs2_transform_point_to_point(color_to_depth_extrinsics, depth_point)
            
            # 将3D点投影到RGB图像平面
            rgb_pixel = rs.rs2_project_point_to_pixel(rgb_intrinsics, rgb_point)
            
            u_r = rgb_pixel[0]
            v_r = rgb_pixel[1]
            
            # 使用双线性插值获取RGB颜色
            if 0 <= u_r < rgb_intrinsics.width - 1 and 0 <= v_r < rgb_intrinsics.height - 1:
                x1, y1 = int(u_r), int(v_r)
                x2, y2 = x1 + 1, y1 + 1
                
                # 双线性插值权重
                dx, dy = u_r - x1, v_r - y1
                
                # 获取四个相邻像素
                p11 = rgb_image[y1, x1]
                p12 = rgb_image[y1, x2]
                p21 = rgb_image[y2, x1]
                p22 = rgb_image[y2, x2]
                
                # 双线性插值
                aligned_rgb[v_d, u_d] = (
                    p11 * (1 - dx) * (1 - dy) +
                    p12 * dx * (1 - dy) +
                    p21 * (1 - dx) * dy +
                    p22 * dx * dy
                )
            elif 0 <= u_r < rgb_intrinsics.width and 0 <= v_r < rgb_intrinsics.height:
                # 如果在边界上，使用最近邻插值
                aligned_rgb[v_d, u_d] = rgb_image[int(v_r), int(u_r)]
    
    return aligned_rgb

if __name__ == "__main__":
    align_images_using_intrinsics(
        depth_path="outputs/images/observation.depths.cam_left/episode_000000/frame_000030.png",
        rgb_path="outputs/images/observation.images.cam_left/episode_000000/frame_000030.png", 
        output_dir="./aligned_output"
    )