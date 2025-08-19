import cv2
import numpy as np
import argparse
import os

class RectangleDetector:
    def __init__(self):
        """初始化矩形检测器"""
        pass
    
    def rgb_to_grayscale(self, image):
        """
        步骤1: RGB转灰度
        将彩色图像转换为灰度图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return gray
    
    def gaussian_smoothing(self, image, kernel_size=(5, 5), sigma=1.0):
        """
        步骤2: 高斯平滑滤波
        使用高斯滤波器减少噪声
        """
        smoothed = cv2.GaussianBlur(image, kernel_size, sigma)
        return smoothed
    
    def edge_detection(self, image, low_threshold=50, high_threshold=150):
        """
        步骤3: 边缘检测
        使用Canny算法检测边缘
        """
        edges = cv2.Canny(image, low_threshold, high_threshold)
        return edges
    
    def detect_closed_contours(self, edges):
        """
        步骤4: 封闭图形检测
        检测边缘图像中的封闭轮廓
        """
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def rectangle_detection(self, contours, min_area=1000, aspect_ratio_range=(0.5, 2.0)):
        """
        步骤5: 矩形检测
        从轮廓中筛选出矩形
        """
        rectangles = []
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # 获取最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            # 计算矩形的宽高比
            width = rect[1][0]
            height = rect[1][1]
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                    rectangles.append({
                        'contour': contour,
                        'box': box,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'center': rect[0]
                    })
        
        return rectangles
    
    def filter_rectangles(self, rectangles, min_area_ratio=0.01, max_area_ratio=0.8):
        """
        步骤6: 滤波
        根据面积比例和其他条件过滤矩形
        """
        if not rectangles:
            return []
        
        # 按面积排序
        rectangles.sort(key=lambda x: x['area'], reverse=True)
        
        filtered_rectangles = []
        for rect in rectangles:
            # 这里可以添加更多的过滤条件
            # 例如：检查矩形的角度、位置等
            filtered_rectangles.append(rect)
        
        return filtered_rectangles
    
    def process_frame(self, frame, draw_results=True):
        """
        处理单个视频帧
        """
        if frame is None:
            return None, []
        
        # 创建结果图像的副本
        result_frame = frame.copy()
        
        # 步骤1: RGB转灰度
        gray = self.rgb_to_grayscale(frame)
        
        # 步骤2: 高斯平滑滤波
        smoothed = self.gaussian_smoothing(gray)
        
        # 步骤3: 边缘检测
        edges = self.edge_detection(smoothed)
        
        # 步骤4: 封闭图形检测
        contours = self.detect_closed_contours(edges)
        
        # 步骤5: 矩形检测
        rectangles = self.rectangle_detection(contours)
        
        # 步骤6: 滤波
        filtered_rectangles = self.filter_rectangles(rectangles)
        
        # 在结果图像上绘制检测到的矩形
        if draw_results:
            for i, rect in enumerate(filtered_rectangles):
                # 绘制矩形边框
                cv2.drawContours(result_frame, [rect['box']], 0, (0, 255, 0), 2)
                
                # 添加标签
                center = rect['center']
                label = f"Rect {i+1}"
                cv2.putText(result_frame, label, 
                           (int(center[0]), int(center[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result_frame, filtered_rectangles
    
    def process_image(self, image_path, output_path=None):
        """
        完整的图像处理流程
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        # 创建结果图像的副本
        result_image = image.copy()
        
        # 步骤1: RGB转灰度
        print("步骤1: RGB转灰度...")
        gray = self.rgb_to_grayscale(image)
        
        # 步骤2: 高斯平滑滤波
        print("步骤2: 高斯平滑滤波...")
        smoothed = self.gaussian_smoothing(gray)
        
        # 步骤3: 边缘检测
        print("步骤3: 边缘检测...")
        edges = self.edge_detection(smoothed)
        
        # 步骤4: 封闭图形检测
        print("步骤4: 封闭图形检测...")
        contours = self.detect_closed_contours(edges)
        
        # 步骤5: 矩形检测
        print("步骤5: 矩形检测...")
        rectangles = self.rectangle_detection(contours)
        
        # 步骤6: 滤波
        print("步骤6: 滤波...")
        filtered_rectangles = self.filter_rectangles(rectangles)
        
        # 在结果图像上绘制检测到的矩形
        for i, rect in enumerate(filtered_rectangles):
            # 绘制矩形边框
            cv2.drawContours(result_image, [rect['box']], 0, (0, 255, 0), 2)
            
            # 添加标签
            center = rect['center']
            label = f"Rect {i+1}"
            cv2.putText(result_image, label, 
                       (int(center[0]), int(center[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 创建处理过程的可视化
        process_images = {
            'Original': image,
            'Grayscale': cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            'Smoothed': cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR),
            'Edges': cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
            'Result': result_image
        }
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"结果已保存到: {output_path}")
        
        print(f"检测到 {len(filtered_rectangles)} 个矩形")
        
        return result_image, filtered_rectangles, process_images
    
    def process_video(self, video_path, output_path=None, show_live=True, save_frames=False):
        """
        处理视频文件
        """
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息:")
        print(f"  - 分辨率: {width}x{height}")
        print(f"  - 帧率: {fps} FPS")
        print(f"  - 总帧数: {total_frames}")
        
        # 设置输出视频
        output_video = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_stats = []
        
        print("开始处理视频...")
        print("按 'q' 键退出，按 's' 键保存当前帧")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 处理当前帧
            result_frame, rectangles = self.process_frame(frame, draw_results=True)
            
            # 添加帧信息
            cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, f"Detected: {len(rectangles)} rectangles", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 记录检测统计
            detection_stats.append(len(rectangles))
            
            # 显示实时结果
            if show_live:
                cv2.imshow('Rectangle Detection', result_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存当前帧
                    cv2.imwrite(f'frame_{frame_count}.jpg', result_frame)
                    print(f"已保存帧 {frame_count}")
            
            # 写入输出视频
            if output_video:
                output_video.write(result_frame)
            
            # 显示进度
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"处理进度: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # 清理资源
        cap.release()
        if output_video:
            output_video.release()
        cv2.destroyAllWindows()
        
        # 输出统计信息
        avg_detections = np.mean(detection_stats)
        max_detections = np.max(detection_stats)
        print(f"\n处理完成!")
        print(f"  - 总帧数: {frame_count}")
        print(f"  - 平均检测数: {avg_detections:.2f}")
        print(f"  - 最大检测数: {max_detections}")
        if output_path:
            print(f"  - 输出视频: {output_path}")
    
    def process_webcam(self, camera_id=0, output_path=None):
        """
        处理实时摄像头输入
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return
        
        print("开始实时检测...")
        print("按 'q' 键退出，按 's' 键保存当前帧")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break
            
            frame_count += 1
            
            # 处理当前帧
            result_frame, rectangles = self.process_frame(frame, draw_results=True)
            
            # 添加帧信息
            cv2.putText(result_frame, f"Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, f"Detected: {len(rectangles)} rectangles", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示结果
            cv2.imshow('Real-time Rectangle Detection', result_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                cv2.imwrite(f'webcam_frame_{frame_count}.jpg', result_frame)
                print(f"已保存帧 {frame_count}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("实时检测结束")
    
    def show_process_images(self, process_images):
        """
        显示处理过程中的各个步骤图像
        """
        for title, img in process_images.items():
            # 调整图像大小以便显示
            height, width = img.shape[:2]
            if width > 800:
                scale = 800 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            
            cv2.imshow(title, img)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='黑色矩形边框检测器')
    parser.add_argument('input', help='输入文件路径（图像或视频）')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('--show-process', action='store_true', help='显示处理过程图像（仅图像模式）')
    parser.add_argument('--webcam', action='store_true', help='使用摄像头进行实时检测')
    parser.add_argument('--camera-id', type=int, default=0, help='摄像头ID（默认: 0）')
    
    args = parser.parse_args()
    
    # 创建检测器实例
    detector = RectangleDetector()
    
    # 检查输入文件是否存在
    if not args.webcam and not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 确定输入类型
    if args.webcam:
        # 实时摄像头模式
        detector.process_webcam(args.camera_id, args.output)
    else:
        # 文件模式
        file_ext = os.path.splitext(args.input)[1].lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            # 图像模式
            result, rectangles, process_images = detector.process_image(args.input, args.output)
            
            if result is not None:
                print("处理完成!")
                
                # 显示处理过程图像
                if args.show_process:
                    detector.show_process_images(process_images)
                else:
                    # 只显示结果图像
                    cv2.imshow('Detection Result', result)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # 视频模式
            detector.process_video(args.input, args.output)
        
        else:
            print(f"不支持的文件格式: {file_ext}")
            print("支持的图像格式: jpg, jpeg, png, bmp, tiff")
            print("支持的视频格式: mp4, avi, mov, mkv")

if __name__ == "__main__":
    # 如果没有命令行参数，显示使用说明
    import sys
    if len(sys.argv) == 1:
        print("使用方法:")
        print("图像检测:")
        print("  python rectangle_detector.py image.jpg [-o result.jpg] [--show-process]")
        print("视频检测:")
        print("  python rectangle_detector.py video.mp4 [-o result.mp4]")
        print("实时检测:")
        print("  python rectangle_detector.py --webcam [--camera-id 0]")
        print("\n示例:")
        print("  python rectangle_detector.py test_image.jpg -o result.jpg --show-process")
        print("  python rectangle_detector.py test_video.mp4 -o result.mp4")
        print("  python rectangle_detector.py --webcam")
        sys.exit(1)
    
    main() 