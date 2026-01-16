import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

def capture_k_frames_from_all_cameras(k_frames=5, save_dir="captures"):
    """
    Capture K frames from all connected RealSense cameras
    
    Args:
        k_frames (int): Number of frames to capture from each camera
        save_dir (str): Directory to save captured images
    """
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create context and get devices
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("No RealSense devices found")
            return
            
        # Create a pipeline for each device
        pipelines = []
        configs = []
        
        print(f"Found {len(devices)} devices")
        
        align_to = rs.stream.depth
        align = rs.align(align_to)
        
        # Initialize all pipelines
        for i, device in enumerate(devices):
            try:
                
                serial = device.get_info(rs.camera_info.serial_number)
                name = device.get_info(rs.camera_info.name)
                
                # if serial[0] == '2':
                #     continue
                
                print(f"\nInitializing device {i+1}: {name} (S/N: {serial})")
                
                
                # Create pipeline and config
                pipeline = rs.pipeline(ctx)
                config = rs.config()
                config.enable_device(serial)
                
                # Configure streams (adjust resolution/fps as needed)
                # config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                if 'f' in serial:   
                    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
                    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
                else:
                    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                    
                
                
                # Start pipeline
                profile = pipeline.start(config)
                device = profile.get_device()
                device.hardware_reset()
                depth_scale = device.first_depth_sensor().get_depth_scale()
                pipelines.append((pipeline, name, serial))
                configs.append(config)
                
                # Warmup - allow some frames to stabilize
                for _ in range(2):
                    pipeline.wait_for_frames()
                    
            except Exception as e:
                print(f"Error initializing device {i+1}: {str(e)}")
        
        # Capture frames from all cameras
        for frame_num in range(1, k_frames + 1):
            print(f"\nCapturing frame {frame_num} of {k_frames}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            for i, (pipeline, name, serial) in enumerate(pipelines):
                try:
                    # Wait for coherent frames
                    frames = pipeline.wait_for_frames()
                    frames = align.process(frames)
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    
                    if not color_frame or not depth_frame:
                        print(f"Device {i+1}: Missing frames")
                        continue
                    
                    # Convert to numpy arrays
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    # Apply colormap to depth (for visualization)
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03), 
                        cv2.COLORMAP_JET
                    )
                    
                    # Save images
                    device_dir = os.path.join(save_dir, f"camera_{i+1}_{serial}")
                    os.makedirs(device_dir, exist_ok=True)
                    
                    color_path = os.path.join(device_dir, f"color_{frame_num}.png")
                    depth_path = os.path.join(device_dir, f"depth_{frame_num}.png")
                    
                    cv2.imwrite(color_path, color_image)
                    cv2.imwrite(depth_path, depth_image)
                    
                    print(f"Device {i+1} ({name}): Saved frame {frame_num}")
                    
                    # Display the images (optional)
                    # cv2.imshow(f'Color {i+1}', color_image)
                    # cv2.imshow(f'Depth {i+1}', depth_colormap)
                    
                except Exception as e:
                    print(f"Error capturing from device {i+1}: {str(e)}")
            
            # Wait for key press between frames (optional)
            # if cv2.waitKey(100) & 0xFF == ord('q'):
            #     break
                
    finally:
        # Cleanup
        print("\nCleaning up...")
        for pipeline, _, _ in pipelines:
            try:
                pipeline.stop()
            except:
                pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Capture 5 frames from each camera and save to 'captures' directory
    capture_k_frames_from_all_cameras(k_frames=5, save_dir="captures")