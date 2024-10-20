from openni import openni2
import cv2
import math
import numpy as np
import threading
import queue
import time  # 导入时间模块
from gui import update_parameters

# 初始参数设置
blur_kernel = 3
canny_low = 50
canny_high = 175
bilateral_d = 9
bilateral_sigma_color = 75
bilateral_sigma_space = 75
display_mode = 0  # 0: 彩色, 1: 灰度, 2: HSV
circularity_min = 0.85
radius_min = 5
radius_max = 30

# 创建队列用于线程间通信
data_queue = queue.Queue()

# 初始化小球信息
ball_info = {
    'position': (0, 0),
    'radius': 0,
    'circularity': 0,
    'time_interval': 0
}
last_time = time.time()

# 实时更新图像的线程函数
def detect_circles_by_contours(frame):
    global blur_kernel, canny_low, canny_high, bilateral_d, bilateral_sigma_color, bilateral_sigma_space, display_mode
    global circularity_min, radius_min, radius_max, ball_info, last_time  # 添加 ball_info 和 last_time

    # 检查并根据用户选择的显示模式调整图像格式
    if display_mode == 1:  # 灰度模式
        if len(frame.shape) == 3:  # 只有在彩色图像时才转换为灰度图像
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif display_mode == 2:  # HSV 模式
        if len(frame.shape) == 3:  # 只有彩色图像能转换为 HSV
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 应用高斯模糊
    if len(frame.shape) == 3:  # 如果是彩色或 HSV 图像
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = frame  # 如果已经是灰度图像，直接使用

    blurred_image = cv2.GaussianBlur(gray_image, (blur_kernel, blur_kernel), 0)
    filtered_image = cv2.bilateralFilter(blurred_image, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    edges = cv2.Canny(filtered_image, canny_low, canny_high)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < math.pi * radius_min * radius_min or area > math.pi * radius_max * radius_max:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > circularity_min:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius_min < radius < radius_max:
                ball_info['position'] = (int(x), int(y))
                ball_info['radius'] = int(radius)
                ball_info['circularity'] = circularity

                # 计算帧率
                current_time = time.time()
                if last_time != current_time:  # 防止除以零
                    ball_info['time_interval'] = 1 / (current_time - last_time)
                last_time = current_time

                center = (int(x), int(y))
                cv2.circle(frame, center, int(radius), (0, 0, 255), 2)

    return frame

def update_callback(new_params):
    global blur_kernel, canny_low, canny_high, bilateral_d, bilateral_sigma_color, bilateral_sigma_space, display_mode

    # 确保参数在合适的范围内
    blur_kernel = max(3, new_params['blur_kernel']) if new_params['blur_kernel'] % 2 == 1 else new_params['blur_kernel'] + 1
    canny_low = max(10, min(100, new_params['canny_low']))
    canny_high = max(100, min(300, new_params['canny_high']))
    bilateral_d = max(1, min(15, new_params['bilateral_d']))
    bilateral_sigma_color = max(50, min(150, new_params['bilateral_sigma_color']))
    bilateral_sigma_space = max(50, min(150, new_params['bilateral_sigma_space']))
    display_mode = new_params['display_mode']

    # 新增参数处理
    circularity_min = max(0.0, min(1.0, new_params['circularity_min']))
    radius_min = max(1, new_params['radius_min'])
    radius_max = max(radius_min, new_params['radius_max'])  # 确保最大半径大于最小半径

if __name__ == "__main__":
    openni2.initialize()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    # 启动 GUI 线程
    gui_thread = threading.Thread(target=update_parameters, args=(update_callback, ball_info))
    gui_thread.start()  # 启动 GUI 线程，但不使用 daemon


    while True:
        ret, color_frame = cap.read()
        if not ret or color_frame is None:
            print("未能成功读取摄像头帧")
            continue

        # 调用小球识别功能
        processed_frame = detect_circles_by_contours(color_frame)
        cv2.imshow('Display', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
