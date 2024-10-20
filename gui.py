import tkinter as tk
from tkinter import ttk

# 初始化参数的默认值
params = {
    'blur_kernel': 3,
    'canny_low': 50,
    'canny_high': 175,
    'bilateral_d': 9,
    'bilateral_sigma_color': 75,
    'bilateral_sigma_space': 75,
    'display_mode': 0,  # 0: 彩色图像, 1: 灰度图像, 2: HSV 图像
    'circularity_min': 0.85,
    'radius_min': 5,
    'radius_max': 30,
}

def update_parameters(callback, ball_info):
    def on_trackbar_change(value, param_name):
        try:
            value = int(value)
            if param_name == 'blur_kernel' and value % 2 == 0:
                value += 1  # 确保高斯模糊内核大小是奇数
            params[param_name] = value
            callback(params)
        except Exception as e:
            print(f"Error updating {param_name}: {e}")

    def on_mode_change():
        params['display_mode'] = mode_var.get()
        callback(params)

    # 创建主窗口
    root = tk.Tk()
    root.title("Parameter Adjustment")

    # 创建滑动条控件
    ttk.Label(root, text="Blur Kernel Size").grid(row=1, column=0, padx=5, pady=0)
    blur_slider = tk.Scale(root, from_=3, to=31, orient=tk.HORIZONTAL, command=lambda v: on_trackbar_change(v, 'blur_kernel'))
    blur_slider.set(params['blur_kernel'])
    blur_slider.grid(row=0, column=0, padx=5, pady=0)

    ttk.Label(root, text="Canny Low Threshold").grid(row=5, column=0, padx=5, pady=0)
    canny_low_slider = tk.Scale(root, from_=10, to=100, orient=tk.HORIZONTAL, command=lambda v: on_trackbar_change(v, 'canny_low'))
    canny_low_slider.set(params['canny_low'])
    canny_low_slider.grid(row=4, column=0, padx=5, pady=0)

    ttk.Label(root, text="Canny High Threshold").grid(row=5, column=1, padx=5, pady=0)
    canny_high_slider = tk.Scale(root, from_=100, to=300, orient=tk.HORIZONTAL, command=lambda v: on_trackbar_change(v, 'canny_high'))
    canny_high_slider.set(params['canny_high'])
    canny_high_slider.grid(row=4, column=1, padx=5, pady=0)

    ttk.Label(root, text="Bilateral Filter d").grid(row=7, column=0, padx=5, pady=0)
    bilateral_d_slider = tk.Scale(root, from_=1, to=15, orient=tk.HORIZONTAL, command=lambda v: on_trackbar_change(v, 'bilateral_d'))
    bilateral_d_slider.set(params['bilateral_d'])
    bilateral_d_slider.grid(row=6, column=0, padx=5, pady=0)

    ttk.Label(root, text="Bilateral Sigma Color").grid(row=7, column=1, padx=5, pady=0)
    bilateral_sigma_color_slider = tk.Scale(root, from_=50, to=150, orient=tk.HORIZONTAL, command=lambda v: on_trackbar_change(v, 'bilateral_sigma_color'))
    bilateral_sigma_color_slider.set(params['bilateral_sigma_color'])
    bilateral_sigma_color_slider.grid(row=6, column=1, padx=5, pady=0)

    ttk.Label(root, text="Bilateral Sigma Space").grid(row=7, column=2, padx=5, pady=0)
    bilateral_sigma_space_slider = tk.Scale(root, from_=50, to=150, orient=tk.HORIZONTAL, command=lambda v: on_trackbar_change(v, 'bilateral_sigma_space'))
    bilateral_sigma_space_slider.set(params['bilateral_sigma_space'])
    bilateral_sigma_space_slider.grid(row=6, column=2, padx=5, pady=0)

    # 添加圆形度和半径参数
    ttk.Label(root, text="Circularity Min").grid(row=1, column=1, padx=5, pady=0)
    circularity_slider = tk.Scale(root, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, command=lambda v: on_trackbar_change(float(v), 'circularity_min'))
    circularity_slider.set(params['circularity_min'])
    circularity_slider.grid(row=0, column=1, padx=5, pady=0)

    ttk.Label(root, text="Radius Min").grid(row=3, column=0, padx=5, pady=0)
    radius_min_slider = tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, command=lambda v: on_trackbar_change(int(v), 'radius_min'))
    radius_min_slider.set(params['radius_min'])
    radius_min_slider.grid(row=2, column=0, padx=5, pady=0)

    ttk.Label(root, text="Radius Max").grid(row=3, column=1, padx=5, pady=0)
    radius_max_slider = tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, command=lambda v: on_trackbar_change(int(v), 'radius_max'))
    radius_max_slider.set(params['radius_max'])
    radius_max_slider.grid(row=2, column=1, padx=5, pady=0)

    # 显示模式选择
    mode_var = tk.IntVar(value=params['display_mode'])
    #ttk.Label(root, text="Display Mode").grid(row=8, column=0, padx=5, pady=0)
    mode_frame = ttk.Frame(root)
    mode_frame.grid(row=8, column=1, padx=5, pady=0)
    ttk.Radiobutton(mode_frame, text='Color', variable=mode_var, value=0, command=on_mode_change).pack(side=tk.LEFT)
    ttk.Radiobutton(mode_frame, text='Gray', variable=mode_var, value=1, command=on_mode_change).pack(side=tk.LEFT)
    ttk.Radiobutton(mode_frame, text='HSV', variable=mode_var, value=2, command=on_mode_change).pack(side=tk.LEFT)

    # 添加新的行来显示小球信息
    info_frame = ttk.Frame(root)
    info_frame.grid(row=9, columnspan=2, pady=10)

    # 创建一个标签来显示小球信息
    info_label = ttk.Label(info_frame, text="Position: (0, 0), Radius: 0, Circularity: 0, Frame Rate: 0 FPS", wraplength=300, anchor='w')
    info_label.grid(row=9, column=1, padx=5)

    # 更新小球信息的函数
    def update_ball_info():
        info_text = (f"Position: {ball_info['position']}, "
                     f"Radius: {ball_info['radius']}, "
                     f"Circularity: {ball_info['circularity']:.2f}, "
                     f"Frame Rate: {ball_info['time_interval']:.2f} FPS")
        info_label.config(text=info_text)  # 更新信息标签
        root.after(10, update_ball_info)  # 每100毫秒更新一次信息

    update_ball_info()  # 启动更新小球信息的循环

    root.mainloop()
