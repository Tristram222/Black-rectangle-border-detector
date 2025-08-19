#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
黑色矩形边框检测器演示程序
支持图像、视频和实时摄像头检测
"""

import cv2
import numpy as np
from rectangle_detector import RectangleDetector

def create_test_video():
    """创建测试视频"""
    print("创建测试视频...")
    
    # 视频参数
    fps = 30
    duration = 5  # 5秒
    width, height = 800, 600
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, fps, (width, height))
    
    for frame_num in range(fps * duration):
        # 创建白色背景
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 添加一些动画效果
        time = frame_num / fps
        offset_x = int(50 * np.sin(time * 2))
        offset_y = int(30 * np.cos(time * 1.5))
        
        # 绘制移动的矩形
        cv2.rectangle(frame, (100 + offset_x, 100 + offset_y), 
                     (300 + offset_x, 300 + offset_y), (0, 0, 0), 3)
        
        # 绘制固定的矩形
        cv2.rectangle(frame, (500, 200), (700, 400), (0, 0, 0), 2)
        
        # 添加噪声
        noise = np.random.normal(0, 5, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(frame)
    
    out.release()
    print("测试视频已创建: test_video.mp4")

def demo_image():
    """图像检测演示"""
    print("\n=== 图像检测演示 ===")
    
    # 创建测试图像
    print("1. 创建测试图像...")
    from create_test_image import create_test_image
    test_image = create_test_image()
    
    # 创建检测器
    print("2. 初始化检测器...")
    detector = RectangleDetector()
    
    # 处理图像
    print("3. 开始图像处理...")
    result, rectangles, process_images = detector.process_image('test_rectangle.jpg', 'result.jpg')
    
    if result is not None:
        print(f"4. 检测完成! 找到 {len(rectangles)} 个矩形")
        
        # 显示处理过程
        print("5. 显示处理过程图像...")
        detector.show_process_images(process_images)
        
        # 显示检测结果
        print("6. 显示最终结果...")
        cv2.imshow('Detection Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 打印检测到的矩形信息
        for i, rect in enumerate(rectangles):
            print(f"矩形 {i+1}:")
            print(f"  - 面积: {rect['area']:.2f}")
            print(f"  - 宽高比: {rect['aspect_ratio']:.2f}")
            print(f"  - 中心点: ({rect['center'][0]:.1f}, {rect['center'][1]:.1f})")
    else:
        print("处理失败!")

def demo_video():
    """视频检测演示"""
    print("\n=== 视频检测演示 ===")
    
    # 创建测试视频
    print("1. 创建测试视频...")
    create_test_video()
    
    # 创建检测器
    print("2. 初始化检测器...")
    detector = RectangleDetector()
    
    # 处理视频
    print("3. 开始视频处理...")
    print("按 'q' 键退出，按 's' 键保存当前帧")
    detector.process_video('test_video.mp4', 'result_video.mp4')

def demo_webcam():
    """实时摄像头检测演示"""
    print("\n=== 实时摄像头检测演示 ===")
    
    # 创建检测器
    print("1. 初始化检测器...")
    detector = RectangleDetector()
    
    # 开始实时检测
    print("2. 开始实时检测...")
    print("按 'q' 键退出，按 's' 键保存当前帧")
    detector.process_webcam()

def main():
    """主演示函数"""
    print("=== 黑色矩形边框检测器演示 ===")
    print("选择演示模式:")
    print("1. 图像检测")
    print("2. 视频检测")
    print("3. 实时摄像头检测")
    print("4. 全部演示")
    print("0. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (0-4): ").strip()
            
            if choice == '0':
                print("退出演示")
                break
            elif choice == '1':
                demo_image()
            elif choice == '2':
                demo_video()
            elif choice == '3':
                demo_webcam()
            elif choice == '4':
                demo_image()
                demo_video()
                demo_webcam()
            else:
                print("无效选择，请输入 0-4")
        except KeyboardInterrupt:
            print("\n演示被中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    main() 