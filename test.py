from openni import openni2
import cv2
import math
import numpy as np
import threading
import queue
import time
import os

class BallDetector:
    def __init__(self):
        self.blur_kernel = 3
        self.canny_low = 50
        self.canny_high = 175
        self.bilateral_d = 9
        self.bilateral_sigma_color = 75
        self.bilateral_sigma_space = 75
        self.circularity_min = 0.80
        self.radius_min = 5
        self.radius_max = 30
        self.x = -1
        self.y = -1
        self.w = -1
        self.h = -1
        self.phx = 0
        self.phy = 0
        

        # HSV 阈值定义
        self.lower_ball_hsv = np.array([0, 0, 200])     # 黄色小球下界
        self.upper_ball_hsv = np.array([80, 255, 255])  # 黄色小球上界

        # 初始化小球信息
        self.ball_info = []

        # 定义保存图像的间隔时间（秒）
        self.save_interval = 1
        self.last_save_time = time.time()

    def preprocess_balls(self, hsv_image):
        roi = hsv_image[self.y:self.y+self.h, self.x:self.x+self.w] 
        # 应用HSV阈值筛选小球
        mask_ball = cv2.inRange(roi, self.lower_ball_hsv, self.upper_ball_hsv)
        mask_ball = cv2.morphologyEx(mask_ball, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算
        mask_ball = cv2.morphologyEx(mask_ball, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))    # 开运算
        
        # 创建黑白图像
        binary_ball = np.zeros_like(mask_ball)
        binary_ball[mask_ball > 0] = 255  # 符合条件的区域设置为白色
    
        # 显示二值图像
        cv2.imshow('Binary Ball', binary_ball)
    
        return binary_ball

    def preprocess_black(self, hsv_image):
        # 转换为HSV色彩空间
        
        
        # 调整黑色的HSV阈值
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 160])  # 降低亮度阈值，捕捉更暗的黑色区域
        
        # 应用HSV阈值筛选黑色区域
        mask_black = cv2.inRange(hsv_image, lower_black, upper_black)
        
        # 调整核大小，形态学操作：去噪和填充
        kernel = np.ones((5, 5), np.uint8)
        mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
        mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
        
        # 创建黑白图像
        binary_black = np.zeros_like(mask_black)
        binary_black[mask_black > 0] = 255  # 符合条件的区域设置为白色
        
        # 显示二值图像
        cv2.imshow('Binary Black', binary_black)
        
        return binary_black


    def detect_balls(self, binary_ball, frame): 
        edges = cv2.Canny(binary_ball, self.canny_low, self.canny_high)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        balls = []

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
                    balls.append((cX+self.x, cY+self.y, radius, circularity, hsv_value))
                    cv2.circle(frame, (cX+self.x, cY+self.y), int(radius), (0, 0, 255), 2)  # 红色

        return balls, frame

    def detect_squares(self, binary_image, frame):
        edges = cv2.Canny(binary_image, 50, 175)  # 已替换为具体数值
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        squares = []
        self.x, self.y, self.w, self.h = -1, -1, -1, -1  # 初始化为无效值

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 80000:
                continue
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                angles = []
                for i in range(4):
                    p1 = approx[i][0]
                    p2 = approx[(i + 1) % 4][0]
                    p0 = approx[(i - 1) % 4][0]
                    v1 = p1 - p0
                    v2 = p2 - p1
                    angle = self.angle_between(v1, v2)
                    angles.append(angle)
                if all(80 < angle < 100 for angle in angles):
                    self.x, self.y, self.w, self.h = cv2.boundingRect(approx)
                    squares.append((self.x, self.y, self.w, self.h))
                    cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)  # 绿色
                    cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)  # 蓝色

        return squares, frame
    
    def angle_between(self, v1, v2):
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0
        angle = np.arccos(dot / (norm1 * norm2))
        return np.degrees(angle)
    
    
    def detect(self, frame):
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        binary_black = self.preprocess_black(hsv_image)
        squares, frame = self.detect_squares(binary_black, frame)
        if self.x == -1 and self.y == -1 and self.w == -1 and self.h == -1:
            print("未检测到方形，跳过处理小球。")
            return frame

        binary_ball = self.preprocess_balls(hsv_image)
        if binary_ball is None:
            print("ROI为空，无法处理小球。")
            return frame
        
        binary_ball = self.preprocess_balls(hsv_image)
        balls, frame = self.detect_balls(binary_ball, frame)

        

        # 处理小球
        self.ball_info = []
        for ball in balls:
            self.ball_info.append({
                'position': (int(ball[0]), int(ball[1])),
                'radius': int(ball[2]),
                'circularity': f"{ball[3]:.2f}",
                'hsv': tuple(int(v) for v in ball[4])
            })
        
        return frame

    def output_ball_info(self):
        #os.system('cls')  # 清空终端（适用于Windows）
        
        # 输出小球信息
                # 简化后的小球输出
        for ball in self.ball_info:
            position = ball['position']
            radius = ball['radius']
            hsv = ball['hsv']
            circularity = ball['circularity']
            print(f"position:{position}")
            print(self.x,self.y,self.w, self.h)
            print(position[0] - self.x - self.w / 2)
            self.phx = int(400*(position[0] - self.x - self.w / 2)/self.w)
            self.phy = int(400*(position[1] - self.y - self.h / 2)/self.h)
            print(f"位置=({self.phx}, {self.phy}), 半径={radius}, HSV={hsv}, 圆形度={circularity}")
            print("-----")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv_frame = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
            hsv_value = hsv_frame[y, x]
            print(f"鼠标点击坐标：({x}, {y}), HSV值：{hsv_value}")

    def process_frame(self, frame):
        # 检测符合HSV的小球区域并显示
        processed_frame = self.detect(frame)
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