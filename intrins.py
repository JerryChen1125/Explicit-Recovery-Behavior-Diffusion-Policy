import pyrealsense2 as rs

# Create a context object
ctx = rs.context()

# Get list of all connected devices
devices = ctx.query_devices()

for device in devices:
    serial_number = device.get_info(rs.camera_info.serial_number)
    print(f"\nDevice with serial number: {serial_number}")
    
    # Create a pipeline for this device
    pipeline = rs.pipeline(ctx)
    config = rs.config()
    config.enable_device(serial_number)
    
    # Start pipeline
    pipeline.start(config)
    
    try:
        # Get active profile
        profile = pipeline.get_active_profile()
        
        # Get all streams from the profile
        streams = profile.get_streams()
        
        for stream in streams:
            if stream.stream_type() == rs.stream.depth:
                # Get depth intrinsics
                intrinsics = stream.as_video_stream_profile().get_intrinsics()
                print(f"Depth Intrinsics: {intrinsics}")
            elif stream.stream_type() == rs.stream.color:
                # Get color intrinsics
                intrinsics = stream.as_video_stream_profile().get_intrinsics()
                print(f"Color Intrinsics: {intrinsics}")
                
    finally:
        # Stop the pipeline
        pipeline.stop()