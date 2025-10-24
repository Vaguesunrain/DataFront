import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- 1. 函数：为信号添加加性高斯白噪声 (AWGN) ---
def add_awgn(signal_iq, target_snr_db):
    # ... (此函数无需改变) ...
    sig_power_watts = np.mean(np.abs(signal_iq)**2)
    target_snr_linear = 10**(target_snr_db / 10)
    noise_power_watts = sig_power_watts / target_snr_linear
    noise_std_dev = np.sqrt(noise_power_watts / 2)
    noise = noise_std_dev * (np.random.randn(len(signal_iq)) + 1j * np.random.randn(len(signal_iq)))
    return signal_iq + noise

# --- 2. 加载I/Q数据 ---
# 信号的原始采样率
F_SIGNAL_ORIGINAL = 20e6 
# 我们模拟的接收机采样率（过采样）
F_SIMULATION = 80e6 # <--- 关键修改：使用更高的采样率
TARGET_SNR_DB = 5

try:
    data_iq_20mhz = np.fromfile('wifi_iq_data.bin', dtype=np.complex64)
    if data_iq_20mhz.size == 0: raise FileNotFoundError
    print(f"成功从文件加载 {len(data_iq_20mhz)} 个样本 (采样率 {F_SIGNAL_ORIGINAL/1e6} MHz)。")
except FileNotFoundError:
    print("文件 'wifi_iq_data.bin' 未找到或为空。将使用随机数据进行演示。")
    data_iq_20mhz = (np.random.randn(4096) + 1j * np.random.randn(4096)) * 0.1

# ==================== 新增部分：上采样 ====================

# --- 3. 将信号从 20 MS/s 上采样到 80 MS/s ---
upsample_factor = int(F_SIMULATION / F_SIGNAL_ORIGINAL)
num_samples_80mhz = len(data_iq_20mhz) * upsample_factor

# 使用 resample 函数进行上采样，它会自动处理插值
data_iq_clean = signal.resample(data_iq_20mhz, num_samples_80mhz)
print(f"上采样后，信号现在有 {len(data_iq_clean)} 个样本 (采样率 {F_SIMULATION/1e6} MHz)。")

# --- 4. 在高采样率下添加噪声 ---
# 注意：现在噪声会分布在整个80MHz带宽内
data_iq_noisy = add_awgn(data_iq_clean, TARGET_SNR_DB)

# --- 5. 重新设计滤波器（使用新的采样率） ---
CUTOFF_FREQ = 9.5e6 # 截止频率不变
NUM_TAPS = 151 # 增加阶数以获得更陡峭的过渡带

# 关键：设计滤波器时，fs 参数必须是新的、更高的采样率
taps = signal.firwin(NUM_TAPS, cutoff=CUTOFF_FREQ, fs=F_SIMULATION, pass_zero='lowpass')

# --- 6. 应用滤波器 ---
data_iq_filtered = signal.lfilter(taps, 1.0, data_iq_noisy)

# ==================== 频谱分析与对比 ====================

# --- 7. DSP 参数 ---
N_FFT = 4096 # 使用更多的FFT点以获得更好的频率分辨率
window = np.hamming(N_FFT)

# --- 8. 频谱计算函数 (无需改变) ---
def calculate_power_spectrum(iq_data, n_fft, win):
    # ... (此函数无需改变) ...
    frame = iq_data[:n_fft]
    if len(frame) < n_fft:
        padded_frame = np.zeros(n_fft, dtype=np.complex64)
        padded_frame[:len(frame)] = frame
        frame = padded_frame
    frame_windowed = frame * win
    fft_shifted = np.fft.fftshift(np.fft.fft(frame_windowed))
    power_spectrum_db = 10 * np.log10(np.abs(fft_shifted)**2 + 1e-12)
    return power_spectrum_db

# --- 9. 计算所有信号的频谱 ---
power_spectrum_db_clean = calculate_power_spectrum(data_iq_clean, N_FFT, window)
power_spectrum_db_noisy = calculate_power_spectrum(data_iq_noisy, N_FFT, window)
power_spectrum_db_filtered = calculate_power_spectrum(data_iq_filtered, N_FFT, window)

# --- 10. 构建频率轴 (使用新的采样率) ---
xf = np.fft.fftshift(np.fft.fftfreq(N_FFT, 1/F_SIMULATION))

# --- 11. 绘制最终的对比图 ---
plt.figure(figsize=(14, 8))
plt.plot(xf / 1e6, power_spectrum_db_noisy, label=f'Noisy Signal (Fs={F_SIMULATION/1e6}MHz)', color='red', alpha=0.5)
plt.plot(xf / 1e6, power_spectrum_db_filtered, label='Filtered Signal', color='green', linewidth=2, linestyle='-')
# 为了清晰，可以先不画蓝色原始信号
# plt.plot(xf / 1e6, power_spectrum_db_clean, label='Upsampled Clean Signal', color='blue', linewidth=1.5, linestyle=':')

plt.title("Visualizing Out-of-Band Noise Suppression via Oversampling")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power (dB)")

# 标记信号带宽
plt.axvspan(-CUTOFF_FREQ/1e6, CUTOFF_FREQ/1e6, color='gray', alpha=0.2, label='Signal Band of Interest')
plt.legend()
plt.grid(True)
plt.ylim(bottom=np.min(power_spectrum_db_noisy) - 10)
plt.xlim(-F_SIMULATION/2/1e6, F_SIMULATION/2/1e6) # 确保X轴显示完整的频谱
plt.show()