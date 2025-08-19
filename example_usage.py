#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
矩形检测器使用示例
展示如何使用RectangleDetector类的各种功能
"""

import cv2
import numpy as np
from rectangle_detector import RectangleDetector

def example_image_processing():
    """图像处理示例"""
    print("=== 图像处理示例 ===")
    
    # 创建检测器
    detector = RectangleDetector()
    
    # 处理图像
    result, rectangles, process_images = detector.process_image('test_rectangle.jpg', 'example_result.jpg')
    
    if result is not None:
        print(f"检测到 {len(rectangles)} 个矩形")
        
        # 显示结果
        cv2.imshow('Example Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def example_video_processing():
    """视频处理示例"""
    print("=== 视频处理示例 ===")
    
    # 创建检测器
    detector = RectangleDetector()
    
    # 处理视频（只处理前几秒用于演示）
    print("开始处理视频...")
    detector.process_video('test_video.mp4', 'example_video_result.mp4')

def example_frame_processing():
    """单帧处理示例"""
    print("=== 单帧处理示例 ===")
    
    # 创建检测器
    detector = RectangleDetector()
    
    # 读取图像
    image = cv2.imread('test_rectangle.jpg')
    if image is not None:
        # 处理单帧
        result_frame, rectangles = detector.process_frame(image)
        
        print(f"检测到 {len(rectangles)} 个矩形")
        
        # 显示结果
        cv2.imshow('Frame Processing Result', result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def example_parameter_tuning():
    """参数调优示例"""
    print("=== 参数调优示例 ===")
    
    # 创建检测器
    detector = RectangleDetector()
    
    # 读取图像
    image = cv2.imread('test_rectangle.jpg')
    if image is None:
        print("无法读取图像")
        return
    
    # 尝试不同的参数
    parameters = [
        {'low_threshold': 30, 'high_threshold': 100, 'min_area': 500},
        {'low_threshold': 50, 'high_threshold': 150, 'min_area': 1000},
        {'low_threshold': 70, 'high_threshold': 200, 'min_area': 2000}
    ]
    
    for i, params in enumerate(parameters):
        print(f"\n参数组合 {i+1}: {params}")
        
        # 使用自定义参数处理
        gray = detector.rgb_to_grayscale(image)
        smoothed = detector.gaussian_smoothing(gray)
        edges = detector.edge_detection(smoothed, 
                                      params['low_threshold'], 
                                      params['high_threshold'])
        contours = detector.detect_closed_contours(edges)
        rectangles = detector.rectangle_detection(contours, 
                                               min_area=params['min_area'])
        filtered_rectangles = detector.filter_rectangles(rectangles)
        
        print(f"检测到 {len(filtered_rectangles)} 个矩形")
        
        # 显示边缘检测结果
        cv2.imshow(f'Edges (Params {i+1})', edges)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """主函数"""
    print("矩形检测器使用示例")
    print("=" * 50)
    
    # 确保有测试图像
    try:
        from create_test_image import create_test_image
        create_test_image()
    except:
        print("创建测试图像失败")
        return
    
    # 运行示例
    try:
        example_image_processing()
        example_frame_processing()
        example_parameter_tuning()
        
        # 视频处理示例（可选）
        print("\n是否运行视频处理示例？(y/n): ", end="")
        choice = input().strip().lower()
        if choice == 'y':
            example_video_processing()
        
        print("\n所有示例完成!")
        
    except Exception as e:
        print(f"运行示例时发生错误: {e}")

if __name__ == "__main__":
    main() 