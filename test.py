import pyrealsense2 as rs
import numpy as np

def get_camera_intrinsics():
    """
    获取RealSense相机的分辨率和内参
    """
    # 创建管道和配置
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 启用深度流
    config.enable_stream(rs.stream.depth)
    # 启用彩色流（如果需要彩色相机内参）
    config.enable_stream(rs.stream.color)
    
    try:
        # 启动管道
        profile = pipeline.start(config)
        
        # 获取深度传感器
        depth_sensor = profile.get_device().first_depth_sensor()
        
        # 获取深度流配置
        depth_profile = profile.get_stream(rs.stream.depth)
        depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        
        print("=" * 50)
        print("深度相机参数:")
        print("=" * 50)
        print(f"分辨率: {depth_intrinsics.width} x {depth_intrinsics.height}")
        print(f"焦距(fx, fy): ({depth_intrinsics.fx:.2f}, {depth_intrinsics.fy:.2f})")
        print(f"主点(cx, cy): ({depth_intrinsics.ppx:.2f}, {depth_intrinsics.ppy:.2f})")
        print(f"畸变模型: {depth_intrinsics.model.name}")
        print(f"畸变系数: {depth_intrinsics.coeffs}")
        
        # 获取彩色流配置（如果可用）
        try:
            color_profile = profile.get_stream(rs.stream.color)
            color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            print("\n" + "=" * 50)
            print("彩色相机参数:")
            print("=" * 50)
            print(f"分辨率: {color_intrinsics.width} x {color_intrinsics.height}")
            print(f"焦距(fx, fy): ({color_intrinsics.fx:.2f}, {color_intrinsics.fy:.2f})")
            print(f"主点(cx, cy): ({color_intrinsics.ppx:.2f}, {color_intrinsics.ppy:.2f})")
            print(f"畸变模型: {color_intrinsics.model.name}")
            print(f"畸变系数: {color_intrinsics.coeffs}")
        except:
            print("\n彩色相机参数获取失败，可能未启用或不可用")
        
        # 获取深度比例（将深度值转换为米）
        depth_scale = depth_sensor.get_depth_scale()
        print(f"\n深度比例（转换为米）: {depth_scale}")
        
        # 创建相机内参矩阵
        depth_matrix = np.array([
            [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
            [0, depth_intrinsics.fy, depth_intrinsics.ppy],
            [0, 0, 1]
        ])
        
        print("\n深度相机内参矩阵:")
        print(depth_matrix)
        
        return {
            'depth': {
                'resolution': (depth_intrinsics.width, depth_intrinsics.height),
                'fx': depth_intrinsics.fx,
                'fy': depth_intrinsics.fy,
                'cx': depth_intrinsics.ppx,
                'cy': depth_intrinsics.ppy,
                'distortion_model': depth_intrinsics.model.name,
                'coeffs': depth_intrinsics.coeffs,
                'matrix': depth_matrix,
                'depth_scale': depth_scale
            },
            'color': {
                'resolution': (color_intrinsics.width, color_intrinsics.height),
                'fx': color_intrinsics.fx,
                'fy': color_intrinsics.fy,
                'cx': color_intrinsics.ppx,
                'cy': color_intrinsics.ppy,
                'distortion_model': color_intrinsics.model.name,
                'coeffs': color_intrinsics.coeffs
            } if 'color_intrinsics' in locals() else None
        }
        
    except Exception as e:
        print(f"获取相机参数时出错: {e}")
        return None
    
    finally:
        # 停止管道
        pipeline.stop()

def get_all_available_configs():
    """
    获取相机支持的所有配置和分辨率
    """
    print("\n" + "=" * 50)
    print("可用配置:")
    print("=" * 50)
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    try:
        # 查找所有设备
        context = rs.context()
        devices = context.query_devices()
        
        for i, device in enumerate(devices):
            print(f"\n设备 {i}: {device.get_info(rs.camera_info.name)}")
            print(f"序列号: {device.get_info(rs.camera_info.serial_number)}")
            
            # 获取所有传感器
            sensors = device.query_sensors()
            for sensor in sensors:
                print(f"\n传感器: {sensor.get_info(rs.camera_info.name)}")
                
                # 获取所有流配置
                stream_profiles = sensor.get_stream_profiles()
                
                resolutions = {}
                for profile in stream_profiles:
                    if profile.is_video_stream_profile():
                        vprofile = profile.as_video_stream_profile()
                        res = (vprofile.width(), vprofile.height())
                        stream_type = profile.stream_type().name
                        
                        if stream_type not in resolutions:
                            resolutions[stream_type] = set()
                        resolutions[stream_type].add(res)
                
                for stream_type, res_set in resolutions.items():
                    print(f"  {stream_type} 支持的分辨率:")
                    for res in sorted(res_set):
                        print(f"    {res[0]} x {res[1]}")
    
    except Exception as e:
        print(f"获取配置时出错: {e}")
    
    finally:
        pipeline.stop()

if __name__ == "__main__":
    print("RealSense相机参数读取器")
    print("=" * 50)
    
    # 获取相机内参
    intrinsics = get_camera_intrinsics()
    
    # 获取所有可用配置
    get_all_available_configs()
    
    if intrinsics:
        print("\n" + "=" * 50)
        print("参数获取完成！")
        print("=" * 50)