# 黑色矩形边框检测器

这是一个使用ai辅助生成的使用OpenCV实现的黑色矩形边框检测程序，可以较好的实现2025电赛E题对于黑色矩形框的识别要求，支持图像、视频和实时摄像头检测。按照以下流程进行图像处理：

1. **RGB转灰度** - 将彩色图像转换为灰度图像
2. **高斯平滑滤波** - 使用高斯滤波器减少噪声
3. **边缘检测** - 使用Canny算法检测边缘
4. **封闭图形检测** - 检测边缘图像中的封闭轮廓
5. **矩形检测** - 从轮廓中筛选出矩形
6. **滤波** - 根据面积比例和其他条件过滤矩形

## 功能特点

- 完整的图像处理流程
- 支持图像、视频和实时摄像头检测
- 可调节的参数设置
- 可视化处理过程
- 支持命令行和程序化调用
- 详细的检测结果输出
- 实时显示检测结果
- 支持保存检测结果

## 安装依赖

```bash
pip install opencv-python numpy
```

## 使用方法

### 1. 命令行使用

#### 图像检测
```bash
# 基本使用
python rectangle_detector.py image.jpg

# 保存结果
python rectangle_detector.py image.jpg -o result.jpg

# 显示处理过程
python rectangle_detector.py image.jpg --show-process

# 完整示例
python rectangle_detector.py image.jpg -o result.jpg --show-process
```

#### 视频检测
```bash
# 处理视频文件
python rectangle_detector.py video.mp4

# 保存处理结果
python rectangle_detector.py video.mp4 -o result.mp4
```

#### 实时摄像头检测
```bash
# 使用默认摄像头
python rectangle_detector.py --webcam

# 使用指定摄像头
python rectangle_detector.py --webcam --camera-id 1
```

### 2. 程序化使用

```python
from rectangle_detector import RectangleDetector

# 创建检测器
detector = RectangleDetector()

# 处理图像
result, rectangles, process_images = detector.process_image('input_image.jpg', 'output.jpg')

# 处理视频
detector.process_video('input_video.mp4', 'output_video.mp4')

# 实时摄像头检测
detector.process_webcam()

# 显示结果
detector.show_process_images(process_images)
```

### 3. 快速演示

```bash
# 运行演示程序
python demo.py
```

演示程序提供以下选项：
- 图像检测演示
- 视频检测演示
- 实时摄像头检测演示
- 全部演示

## 程序结构

- `rectangle_detector.py` - 主要的检测器类
- `create_test_image.py` - 创建测试图像
- `demo.py` - 演示程序
- `README.md` - 说明文档

## 参数说明

### 矩形检测参数

- `min_area`: 最小面积阈值 (默认: 1000)
- `aspect_ratio_range`: 宽高比范围 (默认: (0.5, 2.0))

### 边缘检测参数

- `low_threshold`: Canny算法低阈值 (默认: 50)
- `high_threshold`: Canny算法高阈值 (默认: 150)

### 高斯滤波参数

- `kernel_size`: 核大小 (默认: (5, 5))
- `sigma`: 标准差 (默认: 1.0)

## 支持的文件格式

### 图像格式
- JPG/JPEG
- PNG
- BMP
- TIFF

### 视频格式
- MP4
- AVI
- MOV
- MKV

## 输出结果

程序会输出以下信息：

1. **检测到的矩形数量**
2. **每个矩形的详细信息**：
   - 面积
   - 宽高比
   - 中心点坐标
3. **可视化结果**：
   - 原始图像/视频帧
   - 灰度图像
   - 平滑滤波后的图像
   - 边缘检测结果
   - 最终检测结果

## 实时检测功能

### 视频处理
- 实时显示处理进度
- 支持保存处理结果
- 显示帧信息和检测统计
- 支持按键控制（'q'退出，'s'保存帧）

### 摄像头检测
- 实时显示摄像头画面
- 实时矩形检测和标注
- 支持保存当前帧
- 显示检测统计信息

## 示例输出

### 图像处理
```
步骤1: RGB转灰度...
步骤2: 高斯平滑滤波...
步骤3: 边缘检测...
步骤4: 封闭图形检测...
步骤5: 矩形检测...
步骤6: 滤波...
检测到 4 个矩形
结果已保存到: result.jpg
处理完成!
```

### 视频处理
```
视频信息:
  - 分辨率: 1920x1080
  - 帧率: 30 FPS
  - 总帧数: 150
开始处理视频...
处理进度: 20.0% (30/150)
处理进度: 40.0% (60/150)
...
处理完成!
  - 总帧数: 150
  - 平均检测数: 2.34
  - 最大检测数: 5
  - 输出视频: result_video.mp4
```

## 注意事项

1. 确保输入图像/视频清晰，对比度良好
2. 对于不同的图像，可能需要调整参数以获得最佳效果
3. 程序主要针对白色背景上的黑色矩形边框进行优化
4. 建议在光线充足的环境下拍摄图像/视频以获得更好的检测效果
5. 视频处理可能需要较长时间，请耐心等待
6. 实时检测需要摄像头支持

## 故障排除

如果检测效果不理想，可以尝试：

1. 调整边缘检测的阈值参数
2. 修改最小面积阈值
3. 调整高斯滤波的核大小
4. 检查输入图像/视频的质量和对比度
5. 确保摄像头正常工作（实时检测模式）
6. 检查视频文件格式是否支持

## 性能优化建议

1. 对于实时检测，可以降低图像分辨率以提高处理速度
2. 调整检测参数以平衡准确性和速度
3. 使用GPU加速（如果可用）
4. 对于长视频，可以考虑分段处理 