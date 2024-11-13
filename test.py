#cX, cY是小球物理坐标，单位mm
import cv2
import math
import numpy as np

def preprocess_balls(hsv_image, approx):
    image_points = np.array([point[0] for point in approx], dtype=np.float32)
    image_points = order_points(image_points)
    width = 400
    height = 400

    # 定义目标图像的四个角点
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # 透视变换
    M = cv2.getPerspectiveTransform(image_points, dst_points)
    roi = cv2.warpPerspective(hsv_image, M, (width, height))

    mask_ball = cv2.inRange(roi, np.array([0, 0, 200]), np.array([80, 255, 255]))
    mask_ball = cv2.morphologyEx(mask_ball, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask_ball = cv2.morphologyEx(mask_ball, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary_ball = np.zeros_like(mask_ball)
    binary_ball[mask_ball > 0] = 255
    cv2.imshow('Binary Ball', binary_ball)
    return binary_ball

def preprocess_black(hsv_image):
    mask_black = cv2.inRange(hsv_image, np.array([0, 0, 0]), np.array([180, 255, 160]))
    kernel = np.ones((5, 5), np.uint8)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
    binary_black = np.zeros_like(mask_black)
    binary_black[mask_black > 0] = 255
    cv2.imshow('Binary Black', binary_black)
    return binary_black

def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    angle = np.arccos(dot / (norm1 * norm2))
    return np.degrees(angle)

def detect_squares(binary_image, frame):
    edges = cv2.Canny(binary_image, 50, 175)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_points = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 8000:  # 调整面积阈值
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
                angle = angle_between(v1, v2)
                # 输出angle
                print(f"angle:{angle}")
                angles.append(angle)
            if all(80 < angle < 100 for angle in angles):
                x, y, w, h = cv2.boundingRect(approx)
                approx_points = approx
                # 绘制绿色边框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 绘制蓝色轮廓
                cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)
                # 在四个角点绘制实心红色圆
                for point in approx:
                    cv2.circle(frame, tuple(point[0]), 5, (0, 0, 255), -1)
    return frame, approx_points

def detect_balls(binary_ball, frame, x, y):
    edges = cv2.Canny(binary_ball, 50, 175)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    balls = []
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cX, cY = None, None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < math.pi * 3 ** 2 or area > math.pi * 30 ** 2:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > 0.75:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                radius = int(math.sqrt(area / math.pi))
                hsv_value = hsv_frame[cY - 200, cX - 200]
                balls.append((cX - 200, cY - 200, radius, circularity, hsv_value))
                #cv2.circle(frame, (cX + x, cY + y), radius, (0, 255, 255), 2)
                cv2.circle(binary_ball, (cX, cY), radius, (0, 255, 255), 2)
    return balls, frame, cX, cY

def output_ball_info(balls):
    for ball in balls:
        position = (int(ball[0]), int(ball[1]))  # 物理坐标
        radius = int(ball[2])
        hsv = tuple(int(v) for v in ball[4])
        circularity = f"{ball[3]:.2f}"
        print(f"物理坐标={position}, 半径={radius}, HSV={hsv}, 圆形度={circularity}")
        print("-----")
        
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_frame = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
        hsv_value = hsv_frame[y, x]
        print(f"鼠标点击坐标：({x}, {y}), HSV值：{hsv_value}")

def order_points(pts):
    # 按照左上, 右上, 右下, 左下排序
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect

def process_frame(frame):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    binary_black = preprocess_black(hsv_image)
    frame, approx_points = detect_squares(binary_black, frame)
    if approx_points is None:
        print("未检测到方形，跳过处理小球。")
        cv2.imshow('Display', frame)  # 显示原始彩色图像
        return frame
    binary_ball = preprocess_balls(hsv_image, approx_points)
    balls, frame, cX, cY = detect_balls(binary_ball, frame, 0, 0)
    output_ball_info(balls)
    cv2.imshow('Display', frame)
    cv2.setMouseCallback('Display', mouse_callback, frame)
    return frame

def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
    while True:
        ret, color_frame = cap.read()
        if not ret or color_frame is None:
            print("未能成功读取摄像头帧")
            continue
        process_frame(color_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()