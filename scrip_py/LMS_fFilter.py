import numpy as np
import matplotlib.pyplot as plt

def generate_noise_correlation_signals(point_num,rho):
    """生成相关噪声信号"""
    noise1 = np.random.normal(0, 1, point_num)
    noise2 = rho * noise1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, point_num)
    "normalize to signed 12bit by noise1"
    abs_max = np.max(np.abs(noise1))
    scaling_factor = 2047.0 / abs_max
    noise1 = (noise1 * scaling_factor).astype(np.int16)
    noise2 = (noise2 * scaling_factor).astype(np.int16)
    
    return noise1, noise2

def generate_noise_correlation_signals_byFIR(point_num):
    """通过FIR滤波器生成相关噪声信号"""
    from scipy import signal
    # 生成白噪声
    white_noise = np.random.normal(0, 1, point_num)
    # 设计FIR滤波器
    fir_taps = signal.firwin(101, cutoff=0.1)
    # 生成两个相关信号
    noise1 = signal.lfilter(fir_taps, 1.0, white_noise)
    noise2 = signal.lfilter(fir_taps, 1.0, white_noise)
    return noise1, noise2
def check_correlation(noise1, noise2):
    """计算相关系数"""
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(noise1, label='Noise 1')
    plt.plot(noise2, label='Noise 2')
    plt.title('Generated Correlated Noise Signals')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.scatter(noise1, noise2, alpha=0.5)
    plt.title(f'Scatter Plot (Correlation Coefficient: {rho})')
    plt.xlabel('Noise 1')
    plt.ylabel('Noise 2')
    plt.grid()
    plt.tight_layout()
    plt.show()
    return np.corrcoef(noise1, noise2)[0, 1]

def ramdon_data(point_num, mean=0, std=1):
    """生成随机数据"""
    return np.random.normal(mean, std, point_num)
def data_to_bits(data, threshold=0):
    """将模拟数据转换为二进制位"""
    return (data > threshold).astype(int)

import numpy as np

def Single_tone_Sine_Wave(f0, fs, point_num):
    """
    生成单音正弦波信号，并将其缩放到有符号12-bit的60%幅度。

    参数:
    f0 (float): 信号的频率 (Hz)
    fs (int): 采样频率 (Hz)
    point_num (int): 信号的总点数

    返回值:
    np.ndarray: 包含缩放后的、类型为 np.int16 的信号数组
    """
    # 1. 创建时间数组 t
    t = np.arange(point_num) / fs
    
    # 2. 生成原始浮点信号（范围 [-1.0, 1.0]）
    signal_float = np.sin(2 * np.pi * f0 * t)

    # 3. 确定 12-bit 有符号整数的最大值
    # 2^11 - 1 = 2047
    MAX_12BIT_SIGNED = 2**11 - 1 

    # 4. 计算目标幅度（满幅的 60%）
    TARGET_AMPLITUDE = MAX_12BIT_SIGNED * 0.60 
    
    # 5. 缩放、四舍五入并转换为整数
    # 乘以目标幅度进行缩放，然后使用 .astype(np.int16) 转换为有符号 16-bit 整数
    # 尽管目标是 12-bit，但 Python 中通常使用 np.int16 来存储 12-bit 范围的整数
    signal_scaled = np.round(signal_float * TARGET_AMPLITUDE)
    
    # 确保转换为适合保存的整数类型
    signal_int = signal_scaled.astype(np.int16)

    return signal_int

def mix_signal(A,B):
    return A+B
def show_time_and_freq(signal, fs):
    """显示时域和频域图像"""
    point_num = len(signal)
    t = np.arange(point_num) / fs

    # 计算FFT
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(point_num, 1/fs)

    plt.figure(figsize=(12, 6))

    # 时域图
    plt.subplot(2, 1, 1)
    plt.plot(t * 1e3, signal)
    plt.title('Time Domain Signal')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid()

    # 频域图
    plt.subplot(2, 1, 2)
    plt.plot(fft_freqs[:point_num//2] / 1e3, 20 * np.log10(np.abs(fft_vals[:point_num//2]) / point_num))
    plt.title('Frequency Domain Signal')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()

    plt.tight_layout()
    plt.show()

def LMS_Filter(input_signal,desirable_signal,step,order,working_model):
    weight = np.zeros(order)
    signla_len =  input_signal.length
    output_signal = np.zeros(signla_len)
    en = np.zeros(signla_len)

    for n in range(order ,input_signal.length): # 从第 order 个样本开始，因为需要 order 个历史数据来填充滤波器
        output_signal[n:n+order] = np.dot(weight,input_signal[n:n+order])
        en[n:n+order] = desirable_signal - output_signal
        weight =weight+step* 2*np.dot(en,input_signal)
    if working_model == 0:
        return weight, output_signal
    elif  working_model ==1:
        return weight, en
#main
if __name__ == "__main__":
    point_num = 2000
    rho = 0.8
    noise1, noise2 = generate_noise_correlation_signals(point_num,rho)
    check_correlation(noise1, noise2)
    show_time_and_freq(mix_signal(noise1,Single_tone_Sine_Wave(1e6, 10e6, point_num)),10e6)
    # show_time_and_freq(Single_tone_Sine_Wave(1e6, 10e6, 1024), 10e6)
    # check_correlation(*generate_noise_correlation_signals_byFIR(point_num))
    
    