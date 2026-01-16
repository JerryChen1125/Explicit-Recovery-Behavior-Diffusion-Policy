import pyrealsense2 as rs

# 初始化管道
pipeline = rs.pipeline()
config = rs.config()

# 启用深度流（或彩色流）
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动相机
profile = pipeline.start(config)

# 获取深度流的内参（假设使用深度流）
depth_profile = profile.get_stream(rs.stream.depth)
intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

# 打印内参
print("焦距 (fx, fy):", intrinsics.fx, intrinsics.fy)
print("主点 (ppx, ppy):", intrinsics.ppx, intrinsics.ppy)
print("畸变模型:", intrinsics.model)  # 例如 rs.distortion.none 或 rs.distortion.brown_conrady
print("畸变系数:", intrinsics.coeffs)

# 停止管道
pipeline.stop()