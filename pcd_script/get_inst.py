import pyrealsense2 as rs
import json
import numpy as np

def get_all_realsense_intrinsics():
    """
    获取所有连接的RealSense相机的内参矩阵
    """
    # 创建上下文
    context = rs.context()
    devices = context.query_devices()
    
    intrinsics_dict = {}
    
    print(f"找到 {len(devices)} 个RealSense设备")
    
    for i, device in enumerate(devices):
        try:
            # 获取设备序列号
            serial_number = device.get_info(rs.camera_info.serial_number)
            print(f"\n处理设备 {i+1}: 序列号 {serial_number}")
            
            # 创建管道
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial_number)
            
            # 启用深度流（这会同时启用彩色流）
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # 启动管道
            pipeline.start(config)
            
            # 等待一帧数据以确保获取到内参
            frames = pipeline.wait_for_frames()
            
            # 获取深度和彩色流的内参
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if depth_frame:
                depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                intrinsics_dict[serial_number]["depth"] = intrinsics_to_matrix(depth_intrinsics)
                print(f"深度流内参矩阵获取成功")
            
            if color_frame:
                color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
                intrinsics_dict[serial_number]["color"] = intrinsics_to_matrix(color_intrinsics)
                print(f"彩色流内参矩阵获取成功")
            
            # 停止管道
            pipeline.stop()
            
        except Exception as e:
            print(f"处理设备 {i+1} 时出错: {str(e)}")
            continue
    
    return intrinsics_dict

def intrinsics_to_matrix(intrinsics):
    """
    将RealSense内参对象转换为3x3内参矩阵
    """
    matrix = [
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ]
    return matrix

def save_intrinsics_to_json(intrinsics_dict, filename="pcd_meta/inst.json"):
    """
    将内参字典保存为JSON文件
    """
    # 将numpy数组转换为列表以确保JSON可序列化
    serializable_dict = {}
    for key, matrix in intrinsics_dict.items():
        serializable_dict[key] = matrix
    
    with open(filename, 'w') as f:
        json.dump(serializable_dict, f, indent=4)
    
    print(f"\n内参已保存到: {filename}")

def main():
    try:
        # 获取所有相机的内参
        intrinsics_dict = get_all_realsense_intrinsics()
        
        if intrinsics_dict:
            # 保存到JSON文件
            save_intrinsics_to_json(intrinsics_dict)
            
            # 打印结果
            print("\n内参矩阵结果:")
            for serial, matrix in intrinsics_dict.items():
                print(f"\n{serial}:")
                for row in matrix:
                    print(f"  {row}")
        else:
            print("未找到任何RealSense相机或无法获取内参")
            
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")

if __name__ == "__main__":
    main()