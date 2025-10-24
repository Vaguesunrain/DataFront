import numpy as np
import matplotlib.pyplot as plt

# --- ADC 和信号参数 ---
F_SAMPLE = 80e6  # 采样率
N_POINTS = 4096  # FFT点数
f_in = 10.123e6  # 输入频率 (避免与FFT bin完全重合，更真实)
amplitude = 0.9  # 振幅

# 1. 生成高纯度的模拟输入信号
t = np.arange(N_POINTS) / F_SAMPLE
analog_signal = amplitude * np.sin(2 * np.pi * f_in * t)

# 2. 模拟量化的函数
def quantize(signal, bits):
    scaling_factor = 2**(bits - 1)
    # 将信号缩放到整数码字范围
    quantized_int = np.round(signal * scaling_factor)
    # 再转换回等效的浮点电压值用于分析
    return quantized_int / scaling_factor

# 3. 分别为 8, 12, 16 bit 进行量化
digital_signal_8bit = quantize(analog_signal, 8)
digital_signal_12bit = quantize(analog_signal, 12)
digital_signal_16bit = quantize(analog_signal, 16)

# 4. 频谱分析函数 (使用dB)
def get_spectrum_db(signal, n_points, fs):
    # 加窗可以减少频谱泄漏，看得更清楚
    window = np.hanning(n_points)
    fft_result = np.fft.fft(signal * window)
    # 归一化FFT结果
    fft_normalized = fft_result / (n_points / 2)
    # 计算dB，加一个极小值防止log(0)
    fft_db = 20 * np.log10(np.abs(fft_normalized) + 1e-9)
    fft_freq = np.fft.fftfreq(n_points, 1/fs)
    # 只返回正频率部分
    return fft_freq[:n_points//2], fft_db[:n_points//2]

# 5. 计算每个信号的频谱
freq, spec_8bit = get_spectrum_db(digital_signal_8bit, N_POINTS, F_SAMPLE)
_, spec_12bit = get_spectrum_db(digital_signal_12bit, N_POINTS, F_SAMPLE)
_, spec_16bit = get_spectrum_db(digital_signal_16bit, N_POINTS, F_SAMPLE)

# --- 6. 绘图对比 ---
plt.figure(figsize=(14, 8))

plt.plot(freq / 1e6, spec_16bit, label='16-bit ADC', color='green', alpha=0.8)
plt.plot(freq / 1e6, spec_12bit, label='12-bit ADC', color='blue', alpha=0.8)
plt.plot(freq / 1e6, spec_8bit,  label='8-bit ADC', color='red', alpha=0.8)

plt.title('ADC Quantization Noise Comparison', fontsize=16)
plt.xlabel('Frequency (MHz)', fontsize=12)
plt.ylabel('Magnitude (dBFS)', fontsize=12)
plt.legend()
plt.grid(True)

# <<< --- 关键修改在这里 --- >>>
# 将Y轴下限从-120改为-160，让16-bit的噪声基底可以被看到
plt.ylim(-160, 10) 

plt.show()

# --- 理论验证 ---
# 理论信噪比 (SQNR) 公式: SQNR (dB) = 6.02 * N + 1.76
print(f"理论 8-bit SQNR: {6.02 * 8 + 1.76:.2f} dB")
print(f"理论 12-bit SQNR: {6.02 * 12 + 1.76:.2f} dB")
print(f"理论 16-bit SQNR: {6.02 * 16 + 1.76:.2f} dB")