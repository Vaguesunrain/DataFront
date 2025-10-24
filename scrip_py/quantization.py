import numpy as np
import matplotlib.pyplot as plt

# --- ADC 参数 ---
F_SAMPLE = 80e6  # 采样率
N_BITS = 16      # ADC位数
N_POINTS = 4096  # FFT点数

# 1. 生成一个高纯度的模拟正弦波输入
f_in = 10.1e6 
t = np.arange(N_POINTS) / F_SAMPLE
analog_signal = 0.9 * np.sin(2 * np.pi * f_in * t)

# 2. 模拟理想ADC的量化过程
def quantize(signal, bits):
    scaling_factor = 2**(bits - 1)
    quantized_int = np.round(signal * scaling_factor)
    return quantized_int / scaling_factor

# 3. 得到量化后的数字信号
digital_signal = quantize(analog_signal, N_BITS)
#plot digital_signal and analog_signal for comparison
plt.figure(figsize=(12, 5))
plt.plot(t[:200]*1e6, analog_signal[:200], label='Analog Signal', alpha=0.7)
plt.step(t[:200]*1e6, digital_signal[:200], label='Quantized Signal', where='mid', alpha=0.7)
plt.title('Analog vs Quantized Signal (First 200 Samples)', fontsize=16)
plt.xlabel('Time (µs)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# --- 4. 修正后的频谱分析 ---
#    A. 应用窗函数减少频谱泄漏
window = np.hanning(N_POINTS)
signal_windowed = digital_signal * window

#    B. 计算FFT
fft_result_raw = np.fft.fft(signal_windowed)

#    C. 关键修正：进行归一化
#       我们除以 (N_POINTS / 2) 这样峰值的高度就直接对应于原始信号的幅度
fft_normalized = fft_result_raw / (N_POINTS / 2)

#    D. 计算dB值
#       我们只取正频率部分 (从第0个点到第 N_POINTS/2 个点)
fft_db = 20 * np.log10(np.abs(fft_normalized[:N_POINTS//2]) + 1e-9)

#    E. 构建对应的正频率轴
fft_freq = np.fft.fftfreq(N_POINTS, 1/F_SAMPLE)
freq_axis_mhz = fft_freq[:N_POINTS//2] / 1e6

# --- 5. 绘图 ---
plt.figure(figsize=(12, 7))
plt.plot(freq_axis_mhz, fft_db)

plt.title(f'Correctly Plotted {N_BITS}-bit ADC Spectrum', fontsize=16)
plt.xlabel('Frequency (MHz)', fontsize=12)
plt.ylabel('Magnitude (dBFS)', fontsize=12)
plt.grid(True)
plt.ylim(-160, 10) # 使用我们之前验证过的正确Y轴范围
plt.show()

# 验证峰值幅度
signal_peak_db = 20 * np.log10(0.9) # 理论峰值
print(f"信号理论峰值: {signal_peak_db:.2f} dBFS")
print(f"FFT计算出的峰值: {np.max(fft_db):.2f} dBFS")