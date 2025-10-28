% MATLAB LMS Adaptive Noise Cancellation Demo
% ==========================================

%% 1. 参数定义
clear; clc; close all;

point_num = 20000;
fs = 10e6;          % 采样率 10 MHz
f0 = 1e6;           % 信号频率 1 MHz
rho = 0.99;         % 噪声相关系数
order = 32;         % 滤波器阶数

%% 2. 信号生成 (浮点数, 范围 [-1.0, 1.0])
fprintf('--- Running on ideal floating-point signals ---\n');

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

%% 3. LMS滤波器设置与执行 (浮点数)

% 根据输入功率估算一个合适的步长 (mu)
% 理论上限: mu < 2 / (order * power_of_input)
power_ref = mean(reference_noise.^2);
stability_bound = 2 / (order * power_ref);
mu = stability_bound * 0.1; % 选择上限的10%，这是一个安全且高效的选择

fprintf('Normalized signal power: %.4f\n', power_ref);
fprintf('Calculated step size (mu): %.4f\n', mu);
% 
% % 创建LMS滤波器对象
% lms = dsp.LMSFilter('Length', order, 'StepSize', mu);
% 
% % 执行滤波
% % y是滤波器的输出（预测的噪声），e是误差信号（我们想要的去噪结果）
% [y, e] = lms(reference_noise, primary_signal);
x_input = reference_noise(:);    % 参考输入
d_desired = primary_signal(:);   % 期望信号

signal_len = length(x_input);

% 确保 'order' 和 'mu' 已定义
if ~exist('order', 'var') || ~exist('mu', 'var')
    error('LMSBlock:ParamsMissing', '请确保在运行此代码块前定义了 order 和 mu 变量。');
end

% 初始化状态
lms_weights = zeros(order, 1);
lms_delay_line = zeros(order, 1); % 内部输入延迟线

% 预分配输出变量（与 dsp.LMSFilter 输出变量名保持一致）
y = zeros(signal_len, 1);   % 滤波器输出 (估计的噪声)
e = zeros(signal_len, 1);   % 误差信号 (去噪后的信号)

% 2. 逐样本 LMS 迭代循环
for n = 1:signal_len
    
    % a. 更新输入延迟线 X(n)
    % 将新样本 x_input(n) 推入顶部，形成最新的输入向量
    lms_delay_line = [x_input(n); lms_delay_line(1:end-1)];
    
    % b. 计算滤波器输出 y(n) (估计的噪声)
    y(n) = lms_weights.' * lms_delay_line;
    
    % c. 计算误差 e(n) (去噪后的信号)
    e(n) = d_desired(n) - y(n);
    
    % d. 更新权重 W(n+1)
    % W(n+1) = W(n) + mu * e(n) * X(n)
    lms_weights = lms_weights + mu * e(n) * lms_delay_line;
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
figure('Name', 'LMS Filter Performance (Floating-Point)');
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
legend('Before LMS', 'After LMS');