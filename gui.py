import tkinter as tk
from tkinter import ttk
import numpy as np  # 确保引入 numpy 库

# 初始化参数的默认值
params = {
    'blur_kernel': 3,
    'canny_low': 50,
    'canny_high': 175,
    'bilateral_d': 9,
    'bilateral_sigma_color': 75,
    'bilateral_sigma_space': 75,
    'circularity_min': 0.85,
    'radius_min': 5,
    'radius_max': 30,
    'lower_hsv': np.array([0, 0, 200]),  # 使用 numpy 数组
    'upper_hsv': np.array([100, 255, 255]),  # 使用 numpy 数组
}

def update_callback(config, new_params):
    # 确保参数在合适的范围内
    config.blur_kernel = max(3, new_params['blur_kernel']) if new_params['blur_kernel'] % 2 == 1 else new_params['blur_kernel'] + 1
    config.canny_low = max(10, min(100, new_params['canny_low']))
    config.canny_high = max(100, min(300, new_params['canny_high']))
    config.bilateral_d = max(1, min(15, new_params['bilateral_d']))
    config.bilateral_sigma_color = max(50, min(150, new_params['bilateral_sigma_color']))
    config.bilateral_sigma_space = max(50, min(150, new_params['bilateral_sigma_space']))

    # 更新 lower_hsv 和 upper_hsv，确保它们是 numpy 数组
    config.lower_hsv[:] = np.array(new_params['lower_hsv']) 
    config.upper_hsv[:] = np.array(new_params['upper_hsv'])

    # 新增参数处理
    config.circularity_min = max(0.0, min(1.0, new_params['circularity_min']))
    config.radius_min = max(1, new_params['radius_min'])
    config.radius_max = max(config.radius_min, new_params['radius_max'])

def update_parameters(config, ball_info):
    def on_trackbar_change(value, param_name, index=None):
        try:
            if param_name == 'blur_kernel':
                value = int(value)
                if value % 2 == 0:
                    value += 1  # 确保高斯模糊内核大小是奇数
                params[param_name] = value
            elif param_name in ['lower_hsv', 'upper_hsv']:
                value = int(value)
                params[param_name][index] = value  # 更新 numpy 数组中的值
            elif param_name in ['canny_low', 'canny_high', 'bilateral_d', 'bilateral_sigma_color', 'bilateral_sigma_space', 'radius_min', 'radius_max']:
                value = int(value)
                params[param_name] = value  # 更新其他整数参数
            elif param_name == 'circularity_min':
                value = float(value)
                params[param_name] = value  # 更新浮点数参数
            else:
                value = float(value)  # 如果有其他浮点数参数
                params[param_name] = value
    
            print(f"Updating {param_name} to {value}")
            update_callback(config, params)  # 调用回调函数更新参数
        except Exception as e:
            print(f"Error updating {param_name}: {e}")

    def create_slider(root, text, from_, to, row, column, param_name, index=None, resolution=1):
        ttk.Label(root, text=text).grid(row=row+1, column=column, padx=5, pady=0)
        slider = tk.Scale(root, from_=from_, to=to, orient=tk.HORIZONTAL, resolution=resolution, command=lambda v: on_trackbar_change(v, param_name, index))
        slider.set(params[param_name] if index is None else params[param_name][index])
        slider.grid(row=row, column=column, padx=5, pady=0)
        return slider

    # 创建主窗口
    root = tk.Tk()
    root.title("Parameter Adjustment")

    # 创建滑动条控件
    create_slider(root, "Blur Kernel Size", 3, 31, 0, 0, 'blur_kernel')
    create_slider(root, "Canny Low Threshold", 10, 100, 4, 0, 'canny_low')
    create_slider(root, "Canny High Threshold", 100, 300, 4, 1, 'canny_high')
    create_slider(root, "Bilateral Filter d", 1, 15, 6, 0, 'bilateral_d')
    create_slider(root, "Bilateral Sigma Color", 50, 150, 6, 1, 'bilateral_sigma_color')
    create_slider(root, "Bilateral Sigma Space", 50, 150, 6, 2, 'bilateral_sigma_space')
    create_slider(root, "Circularity Min", 0.0, 1.0, 0, 1, 'circularity_min', resolution=0.01)
    create_slider(root, "Radius Min", 1, 50, 2, 0, 'radius_min')
    create_slider(root, "Radius Max", 1, 50, 2, 1, 'radius_max')
    create_slider(root, "Lower HSV - H", 0, 255, 10, 0, 'lower_hsv', 0)
    create_slider(root, "Lower HSV - S", 0, 255, 10, 1, 'lower_hsv', 1)
    create_slider(root, "Lower HSV - V", 0, 255, 10, 2, 'lower_hsv', 2)
    create_slider(root, "Upper HSV - H", 0, 255, 12, 0, 'upper_hsv', 0)
    create_slider(root, "Upper HSV - S", 0, 255, 12, 1, 'upper_hsv', 1)
    create_slider(root, "Upper HSV - V", 0, 255, 12, 2, 'upper_hsv', 2)

    # 添加新的行来显示小球信息
    info_frame = ttk.Frame(root)
    info_frame.grid(row=9, columnspan=2, pady=10)

    # 创建一个标签来显示小球信息
    info_label = ttk.Label(info_frame, text="Position: (0, 0), Radius: 0, Circularity: 0, Frame Rate: 0 FPS", wraplength=300, anchor='w', width=50)
    info_label.grid(row=9, column=1, padx=5)

    # 更新小球信息的函数
    def update_ball_info():
        hsv_value = ball_info.get('hsv', (0, 0, 0))
        info_text = (f"Position: {ball_info['position']}, "
                     f"Radius: {ball_info['radius']}, "
                     f"Circularity: {ball_info['circularity']:.2f}, "
                     f"HSV: {hsv_value}, "
                     f"Frame Rate: {ball_info['time_interval']:.2f} FPS")
        info_label.config(text=info_text)
        root.after(10, update_ball_info)  # 每100毫秒更新一次信息

    update_ball_info()  # 启动更新小球信息的循环
    print(f"radius_min: {params['radius_min']}, radius_max: {params['radius_max']}")
    root.mainloop()