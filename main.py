from openni import openni2
import cv2
import math
import numpy as np
import threading
import queue
import time
import os
from gui import update_parameters
from sklearn.cluster import KMeans

class BallDetector:
    def __init__(self):
        self.blur_kernel = 3
        self.canny_low = 50
        self.canny_high = 175
        self.bilateral_d = 9
        self.bilateral_sigma_color = 75
        self.bilateral_sigma_space = 75
        self.circularity_min = 0.85
        self.radius_min = 5
        self.radius_max = 30
        self.lower_hsv = np.array([0, 0, 200])
        self.upper_hsv = np.array([100, 255, 255])
        
        # 初始化小球信息
        self.ball_info = {
            'position': (0, 0),
            'radius': 0,
            'circularity': 0,
            'time_interval': 0,
            'hsv': (0, 0, 0)
        }
        
        # 定义保存图像的间隔时间（秒）
        self.save_interval = 1
        self.last_save_time = time.time()
        self.last_time = time.time()

    def preprocess(self, frame):
        # 转换为HSV色彩空间
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 应用HSV阈值筛选
        mask = cv2.inRange(hsv_image, self.lower_hsv, self.upper_hsv)
        # 形态学操作：填充破裂的区域
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 进行闭运算以修复小的断裂
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 移除小的噪声点
        # 创建黑白图像：符合条件的区域变为白色，其余部分为黑色
        binary_output = np.zeros_like(mask)
        binary_output[mask > 0] = 255  # 将符合HSV范围的区域设置为白色

        # ROI - 使用特定的值来定义ROI区域
        x_start = 0    # 左上角 x 坐标
        y_start = 0    # 左上角 y 坐标
        x_end = 640       # 右下角 x 坐标
        y_end = 480      # 右下角 y 坐标

        roi = binary_output[y_start:y_end, x_start:x_end]
        x_offset = x_start
        y_offset = y_start
        
        # 进一步预处理
        blurred_image = cv2.GaussianBlur(roi, (self.blur_kernel, self.blur_kernel), 0)
        filtered_image = cv2.bilateralFilter(blurred_image, self.bilateral_d, self.bilateral_sigma_color, self.bilateral_sigma_space)
        
        return binary_output, filtered_image, x_offset, y_offset

    
    def detect(self, frame):
        binary_output, preprocessed_image, x_offset, y_offset = self.preprocess(frame)
        edges = cv2.Canny(preprocessed_image, self.canny_low, self.canny_high)
    
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        ball_positions = []
        hsv_values = []
    
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < math.pi * self.radius_min * self.radius_min or area > math.pi * self.radius_max * self.radius_max:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > self.circularity_min:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if self.radius_min < radius < self.radius_max:
                    x_global = x + x_offset
                    y_global = y + y_offset
                    center = (int(x_global), int(y_global))
    
                    # 获取小球中心像素的HSV值
                    if 0 <= int(y_global) < hsv_frame.shape[0] and 0 <= int(x_global) < hsv_frame.shape[1]:
                        hsv_value = hsv_frame[int(y_global), int(x_global)]
                        hsv_values.append(hsv_value)
                        ball_positions.append((center, int(radius)))
                    else:
                        continue
    
        # 如果检测到足够的小球
        if len(hsv_values) >= 2:
            hsv_values_np = np.array(hsv_values)
            # 使用KMeans聚类
            kmeans = KMeans(n_clusters=2)
            labels = kmeans.fit_predict(hsv_values_np)
    
            # 统计每个簇的数量
            counts = np.bincount(labels)
            # 找到包含4个小球的簇索引
            blue_cluster_index = np.argmax(counts)
            yellow_cluster_index = 1 - blue_cluster_index
    
            for idx, (center, radius) in enumerate(ball_positions):
                if labels[idx] == yellow_cluster_index:
                    # 标记黄色小球
                    cv2.circle(frame, center, radius, (0, 0, 255), 2)
                    self.ball_info['position'] = center
                    self.ball_info['radius'] = radius
                else:
                    # 标记蓝色小球
                    cv2.circle(frame, center, radius, (255, 0, 0), 2)
        else:
            # 当无法聚类时，标记为未识别
            for (center, radius) in ball_positions:
                cv2.circle(frame, center, radius, (0, 0, 255), 2)
    
        return binary_output, frame
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv_frame = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
            hsv_value = hsv_frame[y, x]
            print(f"鼠标点击坐标：({x}, {y}), HSV值：{hsv_value}")

    def process_frame(self, frame):
        # 检测符合HSV的小球区域并显示
        binary_output, processed_frame = self.detect(frame)
        cv2.imshow('Preprocessed Image', binary_output)
        cv2.imshow('Display', processed_frame)
        cv2.setMouseCallback('Display', self.mouse_callback, frame)        
        # 检查是否需要保存图像
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))
            filename = os.path.join('saves', f"saved_image_{timestamp}.png")
            cv2.imwrite(filename, frame)
            print(f"图像已保存为 {filename}")
            self.last_save_time = current_time

    def run(self):
        openni2.initialize()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            exit()

        # 启动 GUI 线程
        gui_thread = threading.Thread(target=update_parameters, args=(self, self.ball_info))
        gui_thread.start()

        # 创建保存图像的文件夹
        save_folder = 'saves'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        while True:
            ret, color_frame = cap.read()
            if not ret or color_frame is None:
                print("未能成功读取摄像头帧")
                continue

            self.process_frame(color_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# 创建图像处理实例
ball_detector = BallDetector()

if __name__ == "__main__":
    ball_detector.run()