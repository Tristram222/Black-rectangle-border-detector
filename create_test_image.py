import cv2
import numpy as np

def create_test_image():
    """创建一个包含黑色矩形边框的测试图像"""
    # 创建白色背景图像
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # 绘制黑色矩形边框
    # 外矩形
    cv2.rectangle(image, (100, 100), (700, 500), (0, 0, 0), 3)
    
    # 内矩形
    cv2.rectangle(image, (150, 150), (650, 450), (0, 0, 0), 2)
    
    # 添加一些小的矩形
    cv2.rectangle(image, (200, 200), (300, 300), (0, 0, 0), 2)
    cv2.rectangle(image, (500, 200), (600, 300), (0, 0, 0), 2)
    
    # 添加一些噪声（模拟真实图像）
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    # 保存图像
    cv2.imwrite('test_rectangle.jpg', image)
    print("测试图像已创建: test_rectangle.jpg")
    
    return image

if __name__ == "__main__":
    create_test_image() 