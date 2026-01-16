import numpy as np
import cv2
import glob
import os
from matplotlib import pyplot as plt

# Chessboard parameters
BOARD_WIDTH = 10    # Number of inner corners (width)
BOARD_HEIGHT = 7   # Number of inner corners (height)
SQUARE_SIZE = 0.024  # Size of one square in meters (or your preferred unit)

def calibrate_camera(image_paths, camera_name):
    """Calibrate a single camera and return its intrinsics and extrinsics"""
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ..., (8,5,0)
    objp = np.zeros((BOARD_WIDTH * BOARD_HEIGHT, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_WIDTH, 0:BOARD_HEIGHT].T.reshape(-1, 2) * SQUARE_SIZE
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane
    
    # Find chessboard corners
    for fname in image_paths:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (BOARD_WIDTH, BOARD_HEIGHT), None)
        
        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (BOARD_WIDTH, BOARD_HEIGHT), corners2, ret)
            cv2.imshow(f'{camera_name} - Corners', img)
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    
    if not objpoints:
        raise ValueError(f"No chessboard corners found for camera {camera_name}")
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # For extrinsic calibration relative to world (chessboard), we'll use the first image
    # The world coordinate system is defined by the chessboard in the first image
    R, _ = cv2.Rodrigues(rvecs[0])  # Convert rotation vector to matrix
    T = tvecs[0]  # Translation vector
    
    return mtx, dist, R, T

def main():
    # Paths to images from three cameras
    camera_paths = {
        # 'cam1': sorted(glob.glob('captures/camera_1_231522072820/color_*.png')),
        'cam3': sorted(glob.glob('captures/camera_2_f1422097/color_*.png')),
        # 'cam2': sorted(glob.glob('captures/camera_1_317622075882/color_*.png')),
    }
    
    # Verify we have images
    for cam_name, paths in camera_paths.items():
        if not paths:
            raise ValueError(f"No images found for {cam_name}")
        print(f"Found {len(paths)} images for {cam_name}")
    
    # Calibrate each camera
    cameras = {}
    for cam_name, paths in camera_paths.items():
        print(f"\nCalibrating {cam_name}...")
        mtx, dist, R, T = calibrate_camera(paths, cam_name)
        cameras[cam_name] = {
            'K': mtx,       # Intrinsic matrix
            'dist': dist,    # Distortion coefficients
            'R': R,          # Rotation matrix (world to camera)
            'T': T          # Translation vector (world to camera)
        }
        print(f"Calibration complete for {cam_name}")
    
    # Print results
    print("\nExtrinsic parameters (world to camera):")
    for cam_name, params in cameras.items():
        print(f"\n{cam_name}:")
        for key in params:
            
            print(f"{key}:")
            print(params[key])
        # print("Translation vector:")
        # print(params['T'])
    
    # Visualize camera positions (optional)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot world origin (chessboard center)
    ax.scatter(0, 0, 0, c='r', marker='o', s=100, label='World Origin')
    
    # Plot camera positions
    colors = ['b', 'g', 'y']
    for i, (cam_name, params) in enumerate(cameras.items()):
        # Camera position in world coordinates is -R^T * T
        cam_pos = -params['R'].T @ params['T']
        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c=colors[i], marker='^', s=100, label=cam_name)
        
        # Draw camera axes
        axis_length = 0.1
        axes = np.eye(3) * axis_length
        rot_axes = params['R'].T @ axes
        for j in range(3):
            ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                     rot_axes[0, j], rot_axes[1, j], rot_axes[2, j],
                     color=['r', 'g', 'b'][j], length=axis_length)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Positions Relative to World Origin')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()