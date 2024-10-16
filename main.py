from openni import openni2
from openni import _openni2 as c_api
import numpy as np
import cv2
import math
import time  # 引入 time 模块

depth_width = 640
depth_height = 480

# 初始时间
last_detection_time = None  # 记录上次检测到小球的时间

def detect_circles_by_contours(frame):
    global last_detection_time  # 使用全局变量来存储时间

    # 转换为灰度图像
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 应用高斯模糊减少噪声
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    kernel = np.ones((5, 5), np.uint8)
    filtered_image = cv2.bilateralFilter(blurred_image, 9, 75, 75)
    morphed_image = cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, kernel)
    # 使用Canny算子检测边缘
    edges = cv2.Canny(morphed_image, 30, 150)


    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circularity_threshold = 0.85  # 圆形度阈值

    for contour in contours:
        # 过滤掉过小或过大的轮廓，避免无效的计算
        area = cv2.contourArea(contour)
        if area < math.pi * 4 * 4 or area > math.pi * 15 * 15:  # 面积阈值根据具体情况调整
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter * perimeter)

        if circularity > circularity_threshold:
            # 计算最小包围圆并进行进一步过滤
            (x, y), radius = cv2.minEnclosingCircle(contour)

            if radius > 15 or radius <= 4:
                continue

            # 画圆和中心点
            center = (int(x), int(y))
            #cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
            cv2.circle(frame, center, int(radius), (0, 0, 255), 1)
            cv2.circle(frame, center, 3, (0, 0, 255), -1)

            # 输出小球信息
            print(f"({x:.1f}, {y:.1f}) r: {radius:.1f}, circularity: {circularity:.3f}", end='; ')

            # 当前检测时间
            current_time = time.time()

            if last_detection_time is not None:
                # 计算相邻检测的时间间隔
                time_interval = current_time - last_detection_time
                print(f"Time interval: {time_interval * 1000:.2f} ms")

            # 更新上次检测时间
            last_detection_time = current_time
    cv2.imshow('Canny Edges', edges)
    return frame

def mousecallback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(y, x, dpt[y, x])

if __name__ == "__main__":
    openni2.initialize()

    dev = openni2.Device.open_any()
    print(dev.get_device_info())

    depth_stream = dev.create_depth_stream()
    print(depth_stream.get_video_mode())
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                                                   resolutionX=depth_width,
                                                   resolutionY=depth_height, fps=30))
    depth_stream.start()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    cv2.namedWindow('depth')
    cv2.setMouseCallback('depth', mousecallback)

    while True:
        frame = depth_stream.read_frame()

        ret, color_frame = cap.read()
        if not ret or color_frame is None:
            print("未能成功读取摄像头帧")
            continue

        dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([depth_height, depth_width, 2])
        dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
        dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')

        dpt2 = dpt2 * 255
        dpt = dpt1 + dpt2
        dpt = dpt[:, ::-1]
        cv2.imshow('depth', dpt)

        # 在这里调用小球识别功能
        processed_frame = detect_circles_by_contours(color_frame)

        cv2.imshow('color', processed_frame)

        key = cv2.waitKey(1)
        if int(key) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    depth_stream.stop()
    dev.close()