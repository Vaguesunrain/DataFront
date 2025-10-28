% MATLAB VSS-NLMS Adaptive Noise Cancellation - Improved Version
% =============================================================
%% 1. 参数定义 (VSS-NLMS)
clear; clc; close all;
point_num = 20000;
fs = 10e6;          % 采样率 10 MHz
f0 = 1e6;           % 信号频率 1 MHz
rho = 0.99;         % 噪声相关系数
order = 32;         % 滤波器阶数

% --- VSS-NLMS 特有参数 (改进后) ---
alpha = 0.97;       % 步长遗忘因子 (略微降低以提高适应性)
gamma = 1e-3;       % 步长增长调节因子 (增大20倍以提高响应速度)
mu_max = 0.5;       % 步长上限
mu_min = 0.01;      % 步长下限
epsilon = 1e-6;     % 防止分母为零的小常数
% -----------------------------------

%% 2. 信号生成 (浮点数, 范围 [-1.0, 1.0])
fprintf('=== VSS-NLMS 自适应噪声消除 - 改进版 ===\n');

% 时间向量
t = (0:point_num-1)' / fs;

% 生成纯净信号 (s_clean)
s_clean = 0.6 * sin(2 * pi * f0 * t);

% 生成相关噪声 (noise1, noise2)
noise1 = randn(point_num, 1);
noise2 = rho * noise1 + sqrt(1 - rho^2) * randn(point_num, 1);

% 归一化噪声功率
noise1 = noise1 / std(noise1) * 0.4;
noise2 = noise2 / std(noise2) * 0.4;

% 主信号与参考噪声
primary_signal = s_clean + noise1;
reference_noise = noise2;

%% 3. VSS-NLMS 滤波器执行 (改进算法)
x_input = reference_noise(:);    % 参考输入
d_desired = primary_signal(:);   % 期望信号
signal_len = length(x_input);

% 初始化状态
vssnlms_weights = zeros(order, 1);
vssnlms_delay_line = zeros(order, 1);
mu = mu_max * 0.5; % 从中等值开始，提高初始收敛速度

% 预分配输出变量
y = zeros(signal_len, 1);
e = zeros(signal_len, 1);
mu_history = zeros(signal_len, 1);
weight_norm = zeros(signal_len, 1);  % 新增：记录权重范数

% 逐样本 VSS-NLMS 迭代循环
for n = 1:signal_len
    % a. 更新输入延迟线 X(n)
    vssnlms_delay_line = [x_input(n); vssnlms_delay_line(1:end-1)];
    
    % b. 计算滤波器输出 y(n)
    y(n) = vssnlms_weights.' * vssnlms_delay_line;
    
    % c. 计算误差 e(n)
    e(n) = d_desired(n) - y(n);
    
    % d. 计算输入功率 (归一化因子)
    norm_power = vssnlms_delay_line.' * vssnlms_delay_line;
    
    % e. 更新权重 W(n+1) - 使用当前的 mu
    vssnlms_weights = vssnlms_weights + ...
        (mu / (norm_power + epsilon)) * e(n) * vssnlms_delay_line;
    
    % f. (改进) 使用后验误差更新步长 mu(n+1)
    % 后验误差：使用更新后的权重重新计算
    e_post = d_desired(n) - vssnlms_weights.' * vssnlms_delay_line;
    mu_next = alpha * mu + gamma * e_post^2;
    
    % g. 限制 mu 的范围
    mu = max(mu_min, min(mu_max, mu_next));
    
    % 记录性能指标
    mu_history(n) = mu;
    weight_norm(n) = norm(vssnlms_weights);
end

cleaned_signal = e;

%% 4. 性能评估
calculate_snr = @(sig, noise) 10 * log10(mean(sig.^2) / mean(noise.^2));

snr_before = calculate_snr(s_clean, noise1);
residual_noise = cleaned_signal - s_clean;
snr_after = calculate_snr(s_clean, residual_noise);
snr_improvement = snr_after - snr_before;
theoretical_improvement = -10 * log10(1 - rho^2);

% 计算稳态 MSE (取后20%数据)
steady_start = round(0.8 * signal_len);
mse_steady = mean(e(steady_start:end).^2);
excess_mse = mse_steady - mean(residual_noise(steady_start:end).^2);

fprintf('\n=== 性能评估 ===\n');
fprintf('SNR Before:       %.2f dB\n', snr_before);
fprintf('SNR After:        %.2f dB\n', snr_after);
fprintf('SNR Improvement:  %.2f dB\n', snr_improvement);
fprintf('Theoretical Max:  %.2f dB\n', theoretical_improvement);
fprintf('Efficiency:       %.1f%%\n', 100 * snr_improvement / theoretical_improvement);
fprintf('Steady-state MSE: %.2e\n', mse_steady);
fprintf('Excess MSE:       %.2e\n', excess_mse);
fprintf('===================\n');

%% 5. 可视化结果 (改进版)
figure('Name', 'VSS-NLMS Performance Analysis', 'Position', [100 100 1200 800]);

% 5.1 学习曲线 (dB)
subplot(3,2,1);
plot(10*log10(movmean(e.^2, 100)), 'LineWidth', 1.5);
grid on;
title('Learning Curve (Smoothed MSE)');
xlabel('Sample Index');
ylabel('MSE (dB)');
ylim([-60, 0]);

% 5.2 步长 mu 的变化曲线
subplot(3,2,2);
plot(mu_history, 'LineWidth', 1.5);
grid on;
title('Variable Step Size Adaptation');
xlabel('Sample Index');
ylabel('\mu Value');
yline(mu_min, 'r--', 'LineWidth', 1);
yline(mu_max, 'r--', 'LineWidth', 1);
legend('\mu(n)', '\mu_{min}', '\mu_{max}');

% 5.3 权重范数演化
subplot(3,2,3);
plot(weight_norm, 'LineWidth', 1.5);
grid on;
title('Weight Vector Norm Evolution');
xlabel('Sample Index');
ylabel('||W(n)||_2');

% 5.4 频谱对比
subplot(3,2,4);
[pxx_before, f] = pwelch(primary_signal, 2048, [], 2048, fs, 'centered');
[pxx_after, ~] = pwelch(cleaned_signal, 2048, [], 2048, fs, 'centered');
plot(f/1e6, 10*log10(pxx_before), 'LineWidth', 1.5); hold on;
plot(f/1e6, 10*log10(pxx_after), 'LineWidth', 1.5);
grid on;
title('Power Spectral Density');
xlabel('Frequency (MHz)');
ylabel('PSD (dB/Hz)');
legend('Before', 'After');
xlim([0, fs/2/1e6]);

% 5.5 时域信号对比 (前1000样本)
subplot(3,2,5);
n_plot = 1000;
plot(t(1:n_plot)*1e6, primary_signal(1:n_plot), 'LineWidth', 0.8); hold on;
plot(t(1:n_plot)*1e6, cleaned_signal(1:n_plot), 'LineWidth', 1.2);
plot(t(1:n_plot)*1e6, s_clean(1:n_plot), 'k--', 'LineWidth', 1);
grid on;
title('Time Domain Waveforms (First 1000 Samples)');
xlabel('Time (\mus)');
ylabel('Amplitude');
legend('Noisy Input', 'Cleaned Output', 'Clean Signal');

% 5.6 误差分布直方图
subplot(3,2,6);
histogram(e(steady_start:end), 50, 'Normalization', 'pdf');
hold on;
% 手动计算高斯分布 PDF (不需要 Statistics Toolbox)
x_fit = linspace(min(e), max(e), 100);
mu_error = mean(e(steady_start:end));
sigma_error = std(e(steady_start:end));
% 高斯分布公式: f(x) = 1/(sigma*sqrt(2*pi)) * exp(-0.5*((x-mu)/sigma)^2)
pdf_fit = (1/(sigma_error*sqrt(2*pi))) * exp(-0.5*((x_fit-mu_error)/sigma_error).^2);
plot(x_fit, pdf_fit, 'r-', 'LineWidth', 2);
grid on;
title('Steady-State Error Distribution');
xlabel('Error');
ylabel('Probability Density');
legend('Histogram', 'Gaussian Fit');

%% 6. 与标准 NLMS 对比 (可选)
fprintf('\n正在运行标准 NLMS 进行对比...\n');

% 标准 NLMS (固定步长)
mu_fixed = 0.1;
nlms_weights = zeros(order, 1);
nlms_delay_line = zeros(order, 1);
e_nlms = zeros(signal_len, 1);

for n = 1:signal_len
    nlms_delay_line = [x_input(n); nlms_delay_line(1:end-1)];
    y_nlms = nlms_weights.' * nlms_delay_line;
    e_nlms(n) = d_desired(n) - y_nlms;
    norm_power = nlms_delay_line.' * nlms_delay_line;
    nlms_weights = nlms_weights + ...
        (mu_fixed / (norm_power + epsilon)) * e_nlms(n) * nlms_delay_line;
end

cleaned_nlms = e_nlms;
residual_nlms = cleaned_nlms - s_clean;
snr_nlms = calculate_snr(s_clean, residual_nlms);

fprintf('SNR (Fixed NLMS): %.2f dB\n', snr_nlms);
fprintf('VSS-NLMS Gain:    %.2f dB\n', snr_after - snr_nlms);

% 对比学习曲线
figure('Name', 'VSS-NLMS vs Standard NLMS');
plot(10*log10(movmean(e_nlms.^2, 100)), 'LineWidth', 1.5); hold on;
plot(10*log10(movmean(e.^2, 100)), 'LineWidth', 1.5);
grid on;
title('Learning Curve Comparison');
xlabel('Sample Index');
ylabel('MSE (dB)');
legend('Fixed NLMS', 'VSS-NLMS');
ylim([-60, 0]);

fprintf('\n=== 分析完成 ===\n');