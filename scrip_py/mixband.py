import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- 1. 信号和系统参数 ---
# 我们的信号
F_CARRIER_WANTED = 5.8e9
BW = 80e6

# 干扰信号
F_CARRIER_INTERFERER = 6.2e9 

# 两种采样率
F_SAMPLE_BAD = 500e6
F_SAMPLE_GOOD = 440e6

# 模拟参数
F_ANALOG_SIM = 10 * F_SAMPLE_BAD 
N_POINTS_FFT = 8192

# --- 2. 函数：生成一个带限信号并调制到指定载波 ---
def generate_rf_signal(carrier_freq, bandwidth, num_points, fs_sim):
    t = np.arange(num_points) / fs_sim
    baseband = np.random.randn(num_points)
    fir_taps = signal.firwin(201, cutoff=bandwidth/2, fs=fs_sim)
    baseband_filtered = signal.lfilter(fir_taps, 1.0, baseband)
    rf = baseband_filtered * np.cos(2 * np.pi * carrier_freq * t)
    return rf

# --- 3. 生成两个RF信号并相加 ---
num_total_points = N_POINTS_FFT * 20
signal_wanted = generate_rf_signal(F_CARRIER_WANTED, BW, num_total_points, F_ANALOG_SIM)
signal_interferer = generate_rf_signal(F_CARRIER_INTERFERER, BW, num_total_points, F_ANALOG_SIM)

# 模拟空中同时存在两个信号
total_rf_signal = signal_wanted + signal_interferer

# --- 4. 分别用两种采样率进行采样 ---
subsample_ratio_bad = int(F_ANALOG_SIM / F_SAMPLE_BAD)
sampled_signal_bad = total_rf_signal[::subsample_ratio_bad]

subsample_ratio_good = int(F_ANALOG_SIM / F_SAMPLE_GOOD)
sampled_signal_good = total_rf_signal[::subsample_ratio_good]

# --- 5. 可视化对比 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# 图1: 使用“糟糕”的采样率 Fs = 500 MSPS
fft_bad = np.fft.fft(sampled_signal_bad[:N_POINTS_FFT])
freq_bad = np.fft.fftfreq(N_POINTS_FFT, 1 / F_SAMPLE_BAD)
ax1.plot(np.fft.fftshift(freq_bad) / 1e6, np.fft.fftshift(20*np.log10(np.abs(fft_bad))))
ax1.set_title(f'Result with BAD Sampling Rate (Fs = {F_SAMPLE_BAD/1e6} MSPS)', fontsize=14, color='red')
ax1.set_xlabel('Frequency (MHz)')
ax1.set_ylabel('Magnitude (dB)')
ax1.grid(True)
ax1.axvspan(160, 240, color='red', alpha=0.2, label='Corrupted Band [160-240 MHz]')
ax1.axvspan(-240, -160, color='red', alpha=0.2)
ax1.legend()

# 图2: 使用“优秀”的采样率 Fs = 440 MSPS
fft_good = np.fft.fft(sampled_signal_good[:N_POINTS_FFT])
freq_good = np.fft.fftfreq(N_POINTS_FFT, 1 / F_SAMPLE_GOOD)
ax2.plot(np.fft.fftshift(freq_good) / 1e6, np.fft.fftshift(20*np.log10(np.abs(fft_good))))
ax2.set_title(f'Result with GOOD Sampling Rate (Fs = {F_SAMPLE_GOOD/1e6} MSPS)', fontsize=14, color='green')
ax2.set_xlabel('Frequency (MHz)')
ax2.set_ylabel('Magnitude (dB)')
ax2.grid(True)
# 我们的信号混叠位置
ax2.axvspan(40, 120, color='green', alpha=0.2, label='Wanted Signal [40-120 MHz]')
ax2.axvspan(-120, -40, color='green', alpha=0.2)
# 干扰信号混叠位置 (6200 = 14*440 + 40 -> 混叠到40MHz)
ax2.axvspan(0, 80, color='orange', alpha=0.2, label='Interferer [0-80 MHz]')
ax2.axvspan(-80, 0, color='orange', alpha=0.2)
ax2.legend()

fig.tight_layout()
plt.show()