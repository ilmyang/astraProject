import matplotlib.pyplot as plt
import queue

def plot_time_intervals(data_queue):
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots(figsize=(6, 4))  # 设置绘图窗口大小，6x4为适中大小
    line, = ax.plot([], [], lw=2)

    time_intervals = []
    while True:
        try:
            # 尝试从队列中获取数据，设置超时时间避免阻塞
            time_interval = data_queue.get(timeout=1)
            time_intervals.append(time_interval)
            # 只显示最近50个点
            if len(time_intervals) > 50:
                time_intervals.pop(0)

            line.set_xdata(range(len(time_intervals)))
            line.set_ydata(time_intervals)

            ax.relim()
            ax.autoscale_view()

            # 动态设置x轴范围，确保x轴根据索引动态调整
            ax.set_xlim(0, len(time_intervals) - 1)  # X轴范围根据当前有效点数调整
            plt.draw()
            plt.pause(0.01)  # 小暂停来更新绘图
        except queue.Empty:
            continue  # 如果队列为空，继续下一次循环
