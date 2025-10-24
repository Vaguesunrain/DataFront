import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# --- 1. 仿真参数定义 ---
fs = 100e6  # 采样率 (Hz) 100 MSPS
T = 1.0 / fs
N = 8192    # 采样点数 (必须是 2 的幂次方以进行高效 FFT)
t = np.arange(N) * T
fs_nyquist = fs / 2  # 奈奎斯特频率

# 输入信号频率。选择一个非fs/N整数倍的频率，以避免"栅栏效应"的理想情况
# 最好选择一个不与fs/N有公约数的素数倍，以获得更好的频谱分辨率
fin = 12.345e6  # 12.345 MHz 输入正弦波频率

# --- 2. 模拟非线性失真 (谐波产生) ---
# 假设 ADC 传递函数有二次和三次项的非线性：
# V_out = a1*V_in + a2*V_in^2 + a3*V_in^3
A_in = 0.8  # 输入信号幅度 (接近满量程，产生明显失真)
a1 = 1.0    # 线性增益
a2 = 0.05   # 二次非线性系数 (产生二次谐波)
a3 = 0.02   # 三次非线性系数 (产生三次谐波)

# 纯净输入信号
V_in = A_in * np.sin(2.0 * np.pi * fin * t)

# 非线性失真后的信号 (模拟ADC输出)
V_out_analog = a1 * V_in + a2 * V_in**2 + a3 * V_in**3

# --- 3. 模拟量化 (可选, 增加理想量化噪声) ---
# N_bits = 12
# V_LSB = 2.0 / (2**N_bits)  # 假设满量程为 +/- 1V
# V_out_digital = np.round(V_out_analog / V_LSB) * V_LSB
V_out = V_out_analog # 为了集中展示杂散，这里使用连续信号进行 FFT

# --- 4. FFT 频谱分析 ---
# 应用汉宁窗 (Hanning Window) 减少频谱泄漏
window = np.hanning(N)
Y = fft(V_out * window) / N

# 计算频率轴
xf = fftfreq(N, T)[:N//2]

# 计算单边振幅谱 (RMS 幅度，通常用于SFDR/SNR计算)
# 乘以 2 是因为只取了正频率部分，并且 FFT 结果是峰值幅度，需要转换为 RMS
Y_mag_dBc = 20 * np.log10(2 * np.abs(Y[:N//2]) / A_in) # 转换为 dBc

# --- 5. 识别基波和杂散 ---
# 找到基波 (Fundamental) 峰值
fundamental_bin = np.argmax(Y_mag_dBc)
P_fundamental_dBc = Y_mag_dBc[fundamental_bin]

# 排除基波及其附近区域来寻找最大的杂散 (Spur)
# 设置一个排除区域，例如基波左右 10 个 FFT 频点
exclude_range = 10
Y_spur_search = Y_mag_dBc.copy()
Y_spur_search[fundamental_bin - exclude_range : fundamental_bin + exclude_range] = -200 # 设为极低值排除

# 找到最大杂散峰值
spur_bin = np.argmax(Y_spur_search)
P_spur_dBc = Y_spur_search[spur_bin]

# 计算 SFDR
SFDR_dBc = P_fundamental_dBc - P_spur_dBc

# --- 6. 绘图结果 ---
plt.figure(figsize=(12, 6))
plt.plot(xf / 1e6, Y_mag_dBc, label='Spectrum')

# 标记基波
plt.plot(xf[fundamental_bin] / 1e6, P_fundamental_dBc, 'go', 
         label=f'Fundamental ({xf[fundamental_bin]/1e6:.1f} MHz)')

# 标记最大杂散
plt.plot(xf[spur_bin] / 1e6, P_spur_dBc, 'rx', 
         label=f'Max Spur ({xf[spur_bin]/1e6:.1f} MHz)')

# 标记谐波及其混叠
f2_folded = np.abs(2 * fin - np.round(2 * fin / fs) * fs) # 二次谐波混叠
f3_folded = np.abs(3 * fin - np.round(3 * fin / fs) * fs) # 三次谐波混叠
plt.axvline(f2_folded/1e6, color='gray', linestyle='--', alpha=0.6, label=r'$2f_{\text{in}}$ Folded')
plt.axvline(f3_folded/1e6, color='purple', linestyle='--', alpha=0.6, label=r'$3f_{\text{in}}$ Folded')


plt.title(f'Simulated ADC Output Spectrum & SFDR\nSFDR = {SFDR_dBc:.2f} dBc')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Amplitude (dBc)')
plt.xlim(0, fs_nyquist / 1e6) # 只显示 0 到 fs/2 范围
plt.ylim(-100, 0)
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.show()

print(f"采样率 (fs): {fs/1e6:.0f} MSPS")
print(f"奈奎斯特频率 (fs/2): {fs_nyquist/1e6:.0f} MHz")
print(f"输入频率 (fin): {fin/1e6:.3f} MHz")
print(f"基波功率: {P_fundamental_dBc:.2f} dBc")
print(f"最大杂散功率: {P_spur_dBc:.2f} dBc")
print(f"SFDR: {SFDR_dBc:.2f} dBc")