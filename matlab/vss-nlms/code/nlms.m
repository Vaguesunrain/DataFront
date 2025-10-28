% MATLAB NLMS Adaptive Noise Cancellation Demo
% ===========================================

%% 1. 参数定义
clear; clc; close all;

point_num = 20000;
fs = 10e6;          % 采样率 10 MHz
f0 = 1e6;           % 信号频率 1 MHz
rho = 0.99;         % 噪声相关系数
order = 32;         % 滤波器阶数
mu = 0.2;           % NLMS的步长 (通常选择范围在 0 到 2 之间)
epsilon = 1e-6;     % 为防止分母为零而加入的小常数

%% 2. 信号生成 (浮点数, 范围 [-1.0, 1.0])
fprintf('--- Running NLMS on ideal floating-point signals ---\n');

% 时间向量
t = (0:point_num-1)' / fs;

% 生成纯净信号 (s_clean)
s_clean = 0.6 * sin(2 * pi * f0 * t); % 幅度为0.6，在[-1,1]范围内

% 生成相关噪声 (noise1, noise2)
noise1 = randn(point_num, 1);
noise2 = rho * noise1 + sqrt(1 - rho^2) * randn(point_num, 1);

% 归一化噪声功率，使其与信号功率可比
noise1 = noise1 / std(noise1) * 0.4; % 调整噪声标准差为0.4
noise2 = noise2 / std(noise2) * 0.4;

% 主信号 (primary_signal) = 纯净信号 + 噪声1
primary_signal = s_clean + noise1;

% 参考噪声 (reference_noise) = 噪声2
reference_noise = noise2;

%% 3. NLMS滤波器执行 (浮点数)
% 与dsp.LMSFilter对象实现逻辑一致的自定义代码块
fprintf('Using NLMS algorithm with mu = %.2f\n', mu);

x_input = reference_noise(:);    % 参考输入
d_desired = primary_signal(:);   % 期望信号

signal_len = length(x_input);

% 初始化状态
nlms_weights = zeros(order, 1);
nlms_delay_line = zeros(order, 1); % 内部输入延迟线

% 预分配输出变量
y = zeros(signal_len, 1);   % 滤波器输出 (估计的噪声)
e = zeros(signal_len, 1);   % 误差信号 (去噪后的信号)

% 2. 逐样本 NLMS 迭代循环
for n = 1:signal_len
    
    % a. 更新输入延迟线 X(n)
    nlms_delay_line = [x_input(n); nlms_delay_line(1:end-1)];
    
    % b. 计算滤波器输出 y(n) (估计的噪声)
    y(n) = nlms_weights.' * nlms_delay_line;
    
    % c. 计算误差 e(n) (去噪后的信号)
    e(n) = d_desired(n) - y(n);
    
    % d. 更新权重 W(n+1) - 这是与LMS核心区别所在
    % W(n+1) = W(n) + mu / (||X(n)||^2 + epsilon) * e(n) * X(n)
    norm_power = nlms_delay_line.' * nlms_delay_line; % 计算输入向量的L2范数平方
    nlms_weights = nlms_weights + (mu / (norm_power + epsilon)) * e(n) * nlms_delay_line;
end

cleaned_signal = e; % 误差信号就是去噪后的信号

%% 4. 性能评估与可视化 (浮点数)

% 定义一个SNR计算函数
calculate_snr = @(sig, noise) 10 * log10(mean(sig.^2) / mean(noise.^2));

% 计算SNR
snr_before = calculate_snr(s_clean, noise1);
residual_noise = cleaned_signal - s_clean;
snr_after = calculate_snr(s_clean, residual_noise);
snr_improvement = snr_after - snr_before;
theoretical_improvement = -10 * log10(1 - rho^2);

fprintf('\n--- Performance Evaluation ---\n');
fprintf('SNR Before: %.2f dB\n', snr_before);
fprintf('SNR After:  %.2f dB\n', snr_after);
fprintf('SNR Improvement: %.2f dB\n', snr_improvement);
fprintf('Theoretical Max Improvement: %.2f dB\n', theoretical_improvement);
fprintf('===============================\n');

% 绘制学习曲线 (误差平方的移动平均)
figure('Name', 'NLMS Filter Performance (Floating-Point)');
subplot(2,1,1);
plot(10*log10(movmean(e.^2, 100)));
grid on;
title('Learning Curve (Smoothed Squared Error in dB)');
xlabel('Sample Index');
ylabel('MSE (dB)');
legend('Error Signal Power');

% 比较处理前后的信号频谱
subplot(2,1,2);
pwelch(primary_signal, [], [], [], fs, 'centered');
hold on;
pwelch(cleaned_signal, [], [], [], fs, 'centered');
grid on;
title('Power Spectral Density');
legend('Before NLMS', 'After NLMS');