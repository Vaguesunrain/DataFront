import numpy as np
import matplotlib.pyplot as plt

# --- 1. 参数设置 ---
fs = 80e9       # 采样率 (80 GSa/s)
f_signal = 5e9  # 输入信号频率 (5 GHz)
n_points = 256  # 采样点数
T_signal = 1 / f_signal # 信号周期

# --- 2. 生成高分辨率的“模拟”信号 ---
# 为了精确采样，我们用远高于采样率的时间分辨率来生成原始信号
t_analog = np.linspace(0, (n_points-1)/fs, n_points * 16)
signal_i_analog = np.cos(2 * np.pi * f_signal * t_analog) # I路信号 (0度)
signal_q_analog = np.sin(2 * np.pi * f_signal * t_analog) # Q路信号 (90度)

# --- 3. 模拟采样过程 ---
# 理想采样时间点
t_sample = np.arange(n_points) / fs

# a) 理想采样 (Perfect 90-degree skew)
# I路和Q路在同一理想时间点采样各自的0度和90度信号
i_sampled_ideal = np.cos(2 * np.pi * f_signal * t_sample)
q_sampled_ideal = np.sin(2 * np.pi * f_signal * t_sample)

# b) 错误采样 (Timing Skew of 30 degrees)
# 假设I路采样时钟是准的，Q路采样时钟有偏差
# 30度的相位误差 = (30/360) * 信号周期 的时间误差
time_error = (30 / 360) * T_signal
t_sample_skewed_q = t_sample - time_error # Q路采样时刻提前了

i_sampled_skewed = np.cos(2 * np.pi * f_signal * t_sample) # I路仍然在理想时刻采样
q_sampled_skewed = np.sin(2 * np.pi * f_signal * t_sample_skewed_q) # Q路在错误时刻采样

# --- 4. 绘图 ---
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(18, 5))

# 图1: 时域波形对比
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(t_analog * 1e9, signal_i_analog, 'k-', alpha=0.3, label='Analog I (cos)')
ax1.plot(t_analog * 1e9, signal_q_analog, 'r-', alpha=0.3, label='Analog Q (sin)')
ax1.plot(t_sample * 1e9, i_sampled_skewed, 'ko', label='Sampled I')
ax1.plot(t_sample * 1e9, q_sampled_skewed, 'ro', markersize=6, label='Sampled Q (Skewed)')
ax1.set_title('时域波形 (含30度时钟偏斜)', fontsize=14)
ax1.set_xlabel('时间 (ns)', fontsize=12)
ax1.set_ylabel('幅度', fontsize=12)
ax1.legend()
ax1.set_xlim(0, 2 * T_signal * 1e9) # 显示两个信号周期

# 图2: 星座图对比
ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(i_sampled_ideal, q_sampled_ideal, 'bo-', label='理想情况 (90° Skew)')
ax2.plot(i_sampled_skewed, q_sampled_skewed, 'ro-', label='实际情况 (60° Skew)')
ax2.set_title('星座图 (Q vs I)', fontsize=14)
ax2.set_xlabel('I 路幅度', fontsize=12)
ax2.set_ylabel('Q 路幅度', fontsize=12)
ax2.axis('equal')
ax2.legend()

# 图3: 频谱图对比
ax3 = fig.add_subplot(1, 3, 3)
# 理想情况
ideal_complex_signal = i_sampled_ideal + 1j * q_sampled_ideal
fft_ideal = np.fft.fft(ideal_complex_signal)
fft_freq = np.fft.fftfreq(n_points, 1/fs)
ax3.plot(fft_freq / 1e9, 20 * np.log10(np.abs(np.fft.fftshift(fft_ideal))), 'b-', label='理想情况')

# 偏斜情况
skewed_complex_signal = i_sampled_skewed + 1j * q_sampled_skewed
fft_skewed = np.fft.fft(skewed_complex_signal)
ax3.plot(fft_freq / 1e9, 20 * np.log10(np.abs(np.fft.fftshift(fft_skewed))), 'r-', label='含30°时钟偏斜')

ax3.set_title('频谱图 (FFT)', fontsize=14)
ax3.set_xlabel('频率 (GHz)', fontsize=12)
ax3.set_ylabel('幅度 (dB)', fontsize=12)
ax3.set_ylim(-20, 100)
ax3.legend()

plt.tight_layout()
plt.show()