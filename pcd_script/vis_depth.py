import cv2
import numpy as np
import os

def normalize_depth_image(depth_image):
    """
    将深度图像归一化到0-255范围
    """
    # 如果深度图像是浮点型，先转换为合适的范围
    if depth_image.dtype == np.float32 or depth_image.dtype == np.float64:
        # 移除无效值（0或负值）
        valid_depth = depth_image[depth_image > 0]
        if len(valid_depth) > 0:
            min_val = np.min(valid_depth)
            max_val = np.max(valid_depth)
            # 归一化到0-255
            normalized = np.zeros_like(depth_image, dtype=np.uint8)
            mask = depth_image > 0
            normalized[mask] = ((depth_image[mask] - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            return normalized
        else:
            return np.zeros_like(depth_image, dtype=np.uint8)
    else:
        # 如果已经是整数类型，直接归一化
        min_val = np.min(depth_image)
        max_val = np.max(depth_image)
        if max_val - min_val > 0:
            normalized = ((depth_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(depth_image, dtype=np.uint8)
        return normalized

def save_depth_as_visual_image(input_path, output_path=None):
    """
    将深度图像保存为可视化图像
    
    参数:
        input_path: 输入深度图像路径
        output_path: 输出可视化图像路径，如果为None则自动生成
    """
    # 读取图像
    depth_image = cv2.imread(input_path, cv2.IMREAD_ANYDEPTH)
    
    if depth_image is None:
        print(f"错误: 无法读取图像 {input_path}")
        return False
    
    print(f"图像尺寸: {depth_image.shape}")
    print(f"数据类型: {depth_image.dtype}")
    print(f"深度范围: {np.min(depth_image)} - {np.max(depth_image)}")
    
    # 归一化深度图像
    normalized_depth = normalize_depth_image(depth_image)
    
    # 生成输出路径
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_visualized.png"
    
    # 保存图像
    cv2.imwrite(output_path, normalized_depth)
    print(f"可视化图像已保存: {output_path}")
    
    return True

def process_directory(input_dir, output_dir=None):
    """
    处理目录中的所有深度图像
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, "visualized")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    supported_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.exr']
    
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(fmt) for fmt in supported_formats):
            input_path = os.path.join(input_dir, filename)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_visualized.png")
            
            print(f"处理: {filename}")
            save_depth_as_visual_image(input_path, output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='将深度图像保存为可视化图像')
    parser.add_argument('--input', '-i', required=True, help='输入深度图像路径或目录')
    parser.add_argument('--output', '-o', help='输出图像路径或目录')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # 处理单个文件
        save_depth_as_visual_image(args.input, args.output)
    elif os.path.isdir(args.input):
        # 处理目录
        process_directory(args.input, args.output)
    else:
        print(f"错误: 输入路径 {args.input} 不存在")