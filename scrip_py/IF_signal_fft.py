import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import matplotlib.font_manager as fm

# --- 1. 字体设置 (解决中文显示问题) ---
# 尝试设置一个中文字体，请根据您的系统自行调整路径或字体名称
try:
    # 示例：使用 SimHei 或系统默认中文字体
    plt.rcParams['font.family'] = ['SimHei', 'sans-serif'] # 尝试多种字体
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
except:
    pass
# ----------------------------------------

# --- 2. 定义信号和采样参数 ---
# 原始射频 (RF) 信号参数
f_carrier = 5.8e9  # 5.8 GHz 载波频率 (f_c)
f_mod = 20e6       # 20 MHz 调制信号（模拟带宽 B/2）
B = 2 * f_mod      # 信号总带宽 B = 40 MHz

# 欠采样率 (fs)
# 我们选择 f_s = 240 MSPS，满足 f_s > 2*B (40 MHz) 的要求
f_s = 240e6  # 240 MSPS

# 采样点数 N (用于FFT，取 2 的幂次方以提高速度)
N = 2**16
# 总采集时间
T = N / f_s
# 时间序列
t = np.linspace(0, T, N, endpoint=False)

# --- 3. 信号生成 ---
# 信号 S(t) = 载波 * 调制（这里用调幅作为带宽载体）
# 调制信号中心在 f_carrier 上下 f_mod (20 MHz) 形成总带宽 40 MHz
signal_rf = np.sin(2 * np.pi * f_carrier * t) * (1 + 0.5 * np.sin(2 * np.pi * f_mod * t))

# --- 4. 欠采样后的 FFT 分析 ---
# 计算 FFT
yf = fft(signal_rf)
# 计算频率轴
xf = fftfreq(N, 1/f_s)
# 只考虑正频率部分
xf_pos = xf[:N//2]
yf_pos_db = 20 * np.log10(np.abs(yf[:N//2]) / np.max(np.abs(yf[:N//2])))

# --- 5. 计算预期混叠频率 (关键步骤) ---
# 找出最接近 f_carrier 的 f_s 整数倍 m
m = round(f_carrier / f_s)
# m = round(5.8e9 / 240e6) = round(24.166...) = 24
f_alias_center = np.abs(f_carrier - m * f_s)
# f_alias_center = |5.8e9 - 24 * 240e6| = |5.8e9 - 5.76e9| = 40 MHz

f_Nyquist = f_s / 2

print(f"原始载波频率 f_c: {f_carrier / 1e9:.2f} GHz")
print(f"信号带宽 B: {B / 1e6:.2f} MHz")
print(f"采样率 f_s: {f_s / 1e6:.2f} MSPS")
print(f"奈奎斯特频率 f_s/2: {f_Nyquist / 1e6:.2f} MHz")
print(f"混叠中心频率: {f_alias_center / 1e6:.2f} MHz")


# --- 6. 绘制频谱图 ---
plt.figure(figsize=(12, 6))

plt.plot(xf_pos / 1e6, yf_pos_db, label='欠采样后频谱 (Digitized Spectrum)')
plt.title(f'5.8 GHz 信号的低频欠采样恢复 ({f_s / 1e6:.0f} MSPS)')
plt.xlabel('频率 (MHz)')
plt.ylabel('幅度 (dB)')

# 绘制标记线
plt.axvline(f_alias_center / 1e6, color='r', linestyle='--', label=f'混叠中心: {f_alias_center / 1e6:.0f} MHz')
plt.axvline((f_alias_center + B/2) / 1e6, color='g', linestyle=':', label='信号上边带')
plt.axvline((f_alias_center - B/2) / 1e6, color='g', linestyle=':', label='信号下边带')
plt.axvline(f_Nyquist / 1e6, color='k', linestyle='-.', label='奈奎斯特边界 fs/2')


# 设置x轴范围只显示第一个奈奎斯特区
plt.xlim(0, f_Nyquist / 1e6 * 1.05)
plt.ylim(np.max(yf_pos_db) - 60, np.max(yf_pos_db) + 5)
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()