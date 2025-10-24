import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# ==================== 1. 信号和系统参数 (关键修改) ====================
F_CARRIER = 5.8e9
BW = 80e6
F_SAMPLE = 440e6

# <<< 修改1: 降低模拟采样率，使其足够高即可，无需与载波频率过度绑定 >>>
# 10倍于实际采样率已经足以很好地模拟“模拟”信号
F_ANALOG_SIM = 10 * F_SAMPLE 

# <<< 修改2: 增加FFT点数，以获得更长的信号和更多的采样点 >>>
N_POINTS_FFT = 8192 # 从 4096 增加到 8192

# =======================================================================

# --- 2. 生成基带信号 ---
# 乘以一个更大的数，让信号总时长增加
num_total_points = N_POINTS_FFT * 16 # 增加信号长度
t_base = np.arange(num_total_points) / F_ANALOG_SIM
baseband_signal = np.random.randn(len(t_base))
fir_taps = signal.firwin(201, cutoff=BW / 2, fs=F_ANALOG_SIM)
baseband_signal_filtered = signal.lfilter(fir_taps, 1.0, baseband_signal)
baseband_signal_filtered *= 100

# --- 3. 调制到RF ---
rf_signal = baseband_signal_filtered * np.cos(2 * np.pi * F_CARRIER * t_base)

# --- 4. 欠采样 ---
subsample_ratio = int(F_ANALOG_SIM / F_SAMPLE)
sampled_signal = rf_signal[::subsample_ratio]
t_sampled = t_base[::subsample_ratio]

print(f"采样后得到的总点数: {len(sampled_signal)}") # 检查点数是否足够

# --- 5. 可视化结果 ---

## --- 时域图 ---
fig, ax = plt.subplots(figsize=(15, 7))

# 确定绘图的时间范围
# <<< 修改3: 确保绘图索引在新的数组长度范围内 >>>
num_points_to_plot = 400 
if len(t_sampled) < num_points_to_plot:
    num_points_to_plot = len(t_sampled) # 如果点数还是不够，就画出全部

analog_points_limit = int(t_sampled[num_points_to_plot-1] * F_ANALOG_SIM)

ax.plot(t_sampled[:num_points_to_plot] * 1e6, sampled_signal[:num_points_to_plot], label=f'Sampled Signal (at {F_SAMPLE/1e6} MSPS)')
ax.plot(t_base[:analog_points_limit] * 1e6, rf_signal[:analog_points_limit], color='gray', alpha=0.5, label='Analog RF Signal Envelope')
ax.set_title('Time Domain: Overall View with Aligned Axes', fontsize=16)
ax.set_xlabel('Time (us)', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.legend(loc='upper right')
ax.grid(True)

# 创建嵌入图
axins = zoomed_inset_axes(ax, zoom= (F_ANALOG_SIM / F_SAMPLE) * 10 , loc='lower right')
end_point_analog_zoom = int(2.5 / F_CARRIER * F_ANALOG_SIM) # 显示约2.5个载波周期
end_point_sampled_zoom = int(end_point_analog_zoom / subsample_ratio) + 2
axins.plot(t_base[:end_point_analog_zoom] * 1e9, rf_signal[:end_point_analog_zoom], alpha=0.7)
axins.plot(t_sampled[:end_point_sampled_zoom] * 1e9, sampled_signal[:end_point_sampled_zoom], 'o-', color='red')
axins.set_title('Microscopic View')
axins.set_xlabel('Time (ns)')
axins.grid(True)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.show()


## --- 频域图 ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))

# 使用 N_POINTS_FFT 来计算频谱，以获得一致的分辨率
fft_baseband = np.fft.fft(baseband_signal_filtered[:int(N_POINTS_FFT*subsample_ratio*4)])
freq_axis_baseband = np.fft.fftfreq(len(fft_baseband), 1 / F_ANALOG_SIM)
fft_sampled = np.fft.fft(sampled_signal[:N_POINTS_FFT])
freq_axis_sampled = np.fft.fftfreq(len(fft_sampled), 1 / F_SAMPLE)
fft_rf = np.fft.fft(rf_signal[:int(N_POINTS_FFT*subsample_ratio*4)])
freq_axis_rf = np.fft.fftfreq(len(fft_rf), 1 / F_ANALOG_SIM)

# 计算Y轴范围
y_min = np.median(20 * np.log10(np.abs(np.fft.fftshift(fft_sampled)))) - 15
y_max = np.max(20 * np.log10(np.abs(np.fft.fftshift(fft_baseband)))) + 10

# 绘制图1
ax1.plot(np.fft.fftshift(freq_axis_baseband) / 1e6, np.fft.fftshift(20 * np.log10(np.abs(fft_baseband))), color='purple')
ax1.set_title('1. Original Baseband Spectrum (Centered at 0 Hz)', fontsize=14)
ax1.set_xlabel('Frequency (MHz)')
ax1.set_ylabel('Magnitude (dB)')
ax1.set_xlim(-100, 100)
ax1.set_ylim(y_min, y_max)
ax1.axvspan(-BW / 2 / 1e6, BW / 2 / 1e6, color='gray', alpha=0.2, label=f'80 MHz Bandwidth')
ax1.legend()
ax1.grid(True)

# 绘制图2
ax2.plot(freq_axis_rf / 1e9, 20 * np.log10(np.abs(fft_rf)), color='blue')
ax2.set_title('2. RF Spectrum after Modulation (Signal moved to 5.8 GHz)', fontsize=14)
ax2.set_xlabel('Frequency (GHz)')
ax2.set_ylabel('Magnitude (dB)')
ax2.set_xlim(5.7, 5.9)
ax2.grid(True)

# 绘制图3
ax3.plot(np.fft.fftshift(freq_axis_sampled) / 1e6, np.fft.fftshift(20 * np.log10(np.abs(fft_sampled))), color='red')
ax3.set_title('3. Spectrum after Undersampling (Signal aliased down to 80 MHz IF)', fontsize=14)
ax3.set_xlabel('Frequency (MHz)')
ax3.set_ylabel('Magnitude (dB)')
ax3.set_ylim(y_min, y_max)
ax3.axvspan(40, 120, color='green', alpha=0.2, label='Aliased Signal Band [40-120 MHz]')
ax3.axvspan(-120, -40, color='green', alpha=0.2)
ax3.legend()
ax3.grid(True)
fig.tight_layout()
plt.show()

# 验证图代码可以保持不变，此处省略


## --- 频域图 (已更新为三张子图) ---
plt.figure(figsize=(15, 12)) # 增加了图的高度

# ==================== 新增的图: 原始基带信号频谱 ====================
plt.subplot(3, 1, 1)
freq_axis_baseband = np.fft.fftfreq(len(baseband_signal_filtered), 1/F_ANALOG_SIM)
fft_baseband = np.fft.fft(baseband_signal_filtered)
plt.plot(np.fft.fftshift(freq_axis_baseband) / 1e6, np.fft.fftshift(20*np.log10(np.abs(fft_baseband))), color='purple')
plt.title('1. Frequency Domain: Original Baseband Signal Spectrum', fontsize=14)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dB)')
plt.xlim(-100, 100) # 只看基带附近
plt.axvspan(-BW/2/1e6, BW/2/1e6, color='gray', alpha=0.2, label=f'80 MHz Bandwidth')
plt.legend()
plt.grid(True)

# 图2: 原始RF信号的频谱
plt.subplot(3, 1, 2)
freq_axis_rf = np.fft.fftfreq(len(rf_signal), 1/F_ANALOG_SIM)
fft_rf = np.fft.fft(rf_signal)
plt.plot(freq_axis_rf / 1e9, 20*np.log10(np.abs(fft_rf)), color='blue')
plt.title('2. Frequency Domain: RF Signal Spectrum (After Modulation)', fontsize=14)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Magnitude (dB)')
plt.xlim(5.7, 5.9) # 放大到信号所在区域
plt.grid(True)

# 图3: 采样后信号的频谱
plt.subplot(3, 1, 3)
freq_axis_sampled = np.fft.fftfreq(len(sampled_signal), 1/F_SAMPLE)
fft_sampled = np.fft.fft(sampled_signal)
plt.plot(np.fft.fftshift(freq_axis_sampled) / 1e6, np.fft.fftshift(20*np.log10(np.abs(fft_sampled))), color='red')
plt.title('3. Frequency Domain: Sampled Signal Spectrum (After Undersampling)', fontsize=14)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dB)')
# 标记我们理论计算出的混叠频段 [40, 120] MHz
plt.axvspan(40, 120, color='green', alpha=0.2, label='Aliased Signal Band [40-120 MHz]')
plt.axvspan(-120, -40, color='green', alpha=0.2)
plt.legend()
plt.grid(True)

plt.tight_layout() # 自动调整子图间距
plt.show()


freq_base_shifted = np.fft.fftshift(freq_axis_baseband)
fft_base_shifted = np.fft.fftshift(np.abs(fft_baseband))

# 找到0Hz的索引
zero_idx_base = np.argmin(np.abs(freq_base_shifted))
# 提取正负频率
base_pos_freq = freq_base_shifted[zero_idx_base:]
base_pos_spec = fft_base_shifted[zero_idx_base:]
base_neg_freq = freq_base_shifted[:zero_idx_base]
base_neg_spec = fft_base_shifted[:zero_idx_base]

# 2. 获取采样后频谱的两半
freq_sampled_shifted = np.fft.fftshift(freq_axis_sampled)
fft_sampled_shifted = np.fft.fftshift(np.abs(fft_sampled))

# 找到80MHz (中心) 的索引
center_idx_sampled = np.argmin(np.abs(freq_sampled_shifted - 80e6))
# 提取两部分
sampled_lower_half_freq = freq_sampled_shifted[np.where((freq_sampled_shifted >= 40e6) & (freq_sampled_shifted < 80e6))]
sampled_lower_half_spec = fft_sampled_shifted[np.where((freq_sampled_shifted >= 40e6) & (freq_sampled_shifted < 80e6))]
sampled_upper_half_freq = freq_sampled_shifted[np.where((freq_sampled_shifted >= 80e6) & (freq_sampled_shifted <= 120e6))]
sampled_upper_half_spec = fft_sampled_shifted[np.where((freq_sampled_shifted >= 80e6) & (freq_sampled_shifted <= 120e6))]


# 3. 绘图对比
plt.figure(figsize=(15, 7))
plt.title('Verification: Spectrum Unfolding via Undersampling', fontsize=16)

# 绘制采样后频谱的下半部分
plt.plot(sampled_lower_half_freq / 1e6, 20*np.log10(sampled_lower_half_spec), label='Sampled Spectrum [40-80] MHz', color='red', linewidth=4)
# 绘制原始基带频谱的负半部分(翻转到正频率)
plt.plot(-base_neg_freq / 1e6, 20*np.log10(base_neg_spec), label='Original Baseband Spectrum [-40-0] MHz (Flipped)', color='cyan', linestyle='--', linewidth=2)

# 绘制采样后频谱的上半部分
plt.plot(sampled_upper_half_freq / 1e6, 20*np.log10(sampled_upper_half_spec), label='Sampled Spectrum [80-120] MHz', color='green', linewidth=4)
# 绘制原始基带频谱的正半部分
plt.plot(base_pos_freq / 1e6, 20*np.log10(base_pos_spec), label='Original Baseband Spectrum [0-40] MHz', color='magenta', linestyle='--', linewidth=2)

plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True)
plt.xlim(0, 150)
plt.show()