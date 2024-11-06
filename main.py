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
        self.ball_info = []
        
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
            
            # 进一步预处理
            blurred_image = cv2.GaussianBlur(binary_output, (self.blur_kernel, self.blur_kernel), 0)
            filtered_image = cv2.bilateralFilter(blurred_image, self.bilateral_d, self.bilateral_sigma_color, self.bilateral_sigma_space)
            
            return binary_output, filtered_image

    
    def detect(self, frame):
        binary_output, preprocessed_image = self.preprocess(frame)
        edges = cv2.Canny(preprocessed_image, self.canny_low, self.canny_high)
    
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        ball_positions = []
        hsv_values = []
    
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < math.pi * self.radius_min ** 2 or area > math.pi * self.radius_max ** 2:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > self.circularity_min:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    hsv_value = hsv_frame[cY, cX]
                    radius = cv2.minEnclosingCircle(contour)[1]
                    ball_positions.append((cX, cY, radius, circularity, hsv_value))
                    hsv_values.append(hsv_value)
        
        if len(ball_positions) != 5:
            print("检测到的圆形数量不为5，请确保有一个小球和四个标记点。")
            return binary_output, frame
    
        # 使用KMeans进行颜色聚类
        hsv_values_np = np.array(hsv_values)
        kmeans = KMeans(n_clusters=2, n_init=10)
        labels = kmeans.fit_predict(hsv_values_np)
    
        # 统计每个簇的数量
        counts = np.bincount(labels)
        dominant_cluster = np.argmax(counts)
        other_cluster = 1 - dominant_cluster
    
        # 假设数量多的簇为标记点（蓝色）
        colors = ['蓝色', '黄色']
        
        # 分离标记点和小球
        markers = []
        balls = []
        for idx, label in enumerate(labels):
            if label == dominant_cluster:
                markers.append(ball_positions[idx])
                color_label = colors[label]
                cv2.circle(frame, (ball_positions[idx][0], ball_positions[idx][1]), int(ball_positions[idx][2]), (255, 0, 0), 2)
            else:
                balls.append(ball_positions[idx])
                color_label = colors[label]
                cv2.circle(frame, (ball_positions[idx][0], ball_positions[idx][1]), int(ball_positions[idx][2]), (0, 255, 255), 2)
        
        if len(markers) != 4:
            print("检测到的蓝色标记点数量不为4，请检查标记点的颜色和数量。")
            return binary_output, frame
    
        # 获取标记点的像素坐标
        markers_pixel = [(marker[0], marker[1]) for marker in markers]
        self.blue_markers_pixel = markers_pixel
    
        # 定义物理坐标（假设为20cm边长的正方形）
        markers_physical = np.array([
            [0, 0],
            [20, 0],
            [20, 20],
            [0, 20]
        ], dtype='float32')
    
        markers_pixel_np = np.array(markers_pixel, dtype='float32')
        markers_physical_np = markers_physical
    
        # 计算透视变换矩阵
        self.transform_matrix = cv2.getPerspectiveTransform(markers_pixel_np, markers_physical_np)
    
        # 处理小球
        self.ball_info = []
        for ball in balls:
            physical_coords = self.pixel_to_physical((ball[0], ball[1]))
            self.ball_info.append({
                'position': (int(ball[0]), int(ball[1])),
                'radius': int(ball[2]),
                'circularity': f"{ball[3]:.2f}",
                'hsv': tuple(int(v) for v in ball[4]),
                'physical_position': (round(physical_coords[0], 2), round(physical_coords[1], 2))
            })
        
        return binary_output, frame
    
    def pixel_to_physical(self, pixel_coords):
        if self.transform_matrix is None:
            print("转换矩阵尚未计算。")
            return (0, 0)
        
        pixel = np.array([[pixel_coords]], dtype='float32')
        physical = cv2.perspectiveTransform(pixel, self.transform_matrix)
        physical_coords = physical[0][0]
        return physical_coords
    
    def output_ball_info(self):
        os.system('cls')  # 清空终端（适用于Windows）
        for i, ball in enumerate(self.ball_info, 1):
            position = ball['position']
            radius = ball['radius']
            hsv = ball['hsv']
            circularity = ball['circularity']
            physical = ball['physical_position']
            print(f"球{i}: 位置={position}, 半径={radius}, HSV={hsv}, 圆形度={circularity}, 物理位置={physical} cm")
            if i % 5 == 0:
                print("-----")
        if len(self.ball_info) % 5 != 0:
            print("-----")

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
        # 输出小球信息
        self.output_ball_info()
        # 检查是否需要保存图像
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))
            filename = os.path.join('saves', f"saved_image_{timestamp}.png")
            cv2.imwrite(filename, frame)
            self.last_save_time = current_time

    def run(self):
        openni2.initialize()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            exit()

        # 启动 GUI 线程
        gui_thread = threading.Thread(target=update_parameters, args=(self,))
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