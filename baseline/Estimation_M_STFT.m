%% 参数设置与干扰类型选择 
close all; clear; clc;
% 可选干扰类型： 'ISPRJ'、'ISDRJ' 以及 'ISCRJ'
jam_type = 'ISPRJ'; 

if strcmp(jam_type, 'ISPRJ')
    % ----------- ISPRJ 干扰（间歇采样重复转发干扰） ------------
    Tp = 20e-6;                 % 雷达信号脉宽 20us
    B = 100e6;                  % 雷达信号带宽 100MHz
    Kr = B/Tp;                  % 调频斜率
    fs = 2*B;                   % 采样频率 200MHz
    SNR = 20;                   % 信噪比 (dB)
    M = 3;                      % 转发次数
    tao_jam = 1e-6;             % 干扰采样脉宽 1us
    Ts_jam = (M+1) * tao_jam;   % 干扰采样周期
    
    % 生成时间轴及回波信号
    Ts = 1/fs;
    t = 0:Ts:Tp-Ts;
    N_sample = length(t);
    Sig_rec = exp(1i*pi*Kr*(t).^2);
    
    % 生成采样脉冲
    N_tao = round(tao_jam / Ts);
    N_Ts_jam = round(Ts_jam / Ts);
    N_jam = floor(Tp / Ts_jam);
    Sig_pulse = zeros(1, N_sample);
    for ii = 0:N_jam-1
        start_idx = 1 + ii * N_Ts_jam;
        end_idx = min(start_idx + N_tao - 1, N_sample);
        Sig_pulse(start_idx:end_idx) = 1;
    end
    
    % 生成干扰信号：重复转发（除第一路）累加
    Sig_jam_matrix = zeros(M+1, N_sample);
    Sig_jam_matrix(1,:) = Sig_rec .* Sig_pulse;
    for m = 1:M
        delay_samples = m * N_tao;
        Sig_jam_matrix(m+1,:) = circshift(Sig_jam_matrix(1,:), delay_samples);
    end
    Sig_jam = sum(Sig_jam_matrix(2:end,:), 1);
    
elseif strcmp(jam_type, 'ISDRJ')
    % ----------- ISDRJ 干扰（间歇采样直接转发干扰） ------------
    Tp = 10e-6;                 % 雷达信号脉宽 10us
    B = 100e6;                  % 雷达信号带宽 100MHz
    Kr = B/Tp;                  % 调频斜率
    fs = 2*B;                   % 采样频率 200MHz
    SNR = 10;                   % 信噪比 (dB)
    Ts_jam = 2e-6;              % 干扰采样周期 2us
    tao_jam = 1e-6;             % 干扰采样脉宽 1us
    
    Ts = 1/fs;
    t = 0:Ts:Tp-Ts;
    N_sample = length(t);
    Sig_rec = exp(1i*pi*Kr*(t).^2);
    
    % 生成采样脉冲
    N_tao = round(tao_jam / Ts);
    N_Ts_jam = round(Ts_jam / Ts);
    N_jam = floor(Tp / Ts_jam);
    Sig_pulse = zeros(1, N_sample);
    for ii = 0:N_jam-1
        start_idx = 1 + ii * N_Ts_jam;
        end_idx = min(start_idx + N_tao - 1, N_sample);
        Sig_pulse(start_idx:end_idx) = 1;
    end
    
    % 生成干扰信号：直接转发，带1us延迟
    tau_delay = 1e-6;
    delay_samples = round(tau_delay / Ts);
    Sig_jam_ISDRJ = Sig_rec .* Sig_pulse;
    Sig_jam_ISDRJ = circshift(Sig_jam_ISDRJ, delay_samples);
    Sig_jam = Sig_jam_ISDRJ;
    
elseif strcmp(jam_type, 'ISCRJ')
    % ----------- ISCRJ 干扰 ------------
    Tp = 20e-6;                % 雷达信号脉宽 
    B = 100e6;                  % 雷达信号带宽 
    Kr = B / Tp;               % 调频斜率
    Ts_jam = 4e-6;             % 间歇采样干扰的采样周期
    tao_jam = 1e-6;            % 间歇采样干扰的采样脉宽 
    SNR  = 10;
    M = Ts_jam / tao_jam - 1;  % 延拓次数
    N_sample_jam = Tp / Ts_jam;
    R = round(min(M, N_sample_jam));
    fs = 2 * B;                % 采样频率
    Ts = 1 / fs;
    T_sample = Tp;             % 采样时间
    N_sample = ceil(T_sample / Ts);
    N_sample_tao_jam = ceil(tao_jam / Ts);
    N_sample_Ts_jam = ceil(Ts_jam / Ts);
    
    % 时间轴与回波信号（LFM信号）生成
    t = linspace(0, T_sample, N_sample);
    Sig_rec = exp(1i * pi * Kr * (t) .^ 2);
    
    %% 间歇采样循环转发干扰生成
    % 采样脉冲
    Sig_pulse = zeros(1, N_sample);
    N_jam = Tp / Ts_jam;      % 干扰采样次数，即采样脉冲个数
    for ii = 1:N_jam
        Sig_pulse((1 + (ii - 1) * N_sample_Ts_jam):N_sample_tao_jam + (ii - 1) * N_sample_Ts_jam) = 1;
    end
    %% 干扰信号
    Sig_jam = Sig_rec .* Sig_pulse;
    Sig_jam_matrix = zeros(R + 1, N_sample + (R - 1) * N_sample_Ts_jam);
    Sig_jam_matrix(1, 1:N_sample) = Sig_jam;
    Sig_jam_matrix(2, 1:N_sample + N_sample_tao_jam) = [zeros(1, N_sample_tao_jam), Sig_jam];
    for ii = 3:R + 1
        Sig_jam_matrix(ii, 1:N_sample + N_sample_tao_jam + (ii - 2) * (N_sample_Ts_jam + N_sample_tao_jam)) = ...
            [zeros(1, N_sample_Ts_jam + N_sample_tao_jam), Sig_jam_matrix(ii - 1, 1:N_sample + N_sample_tao_jam + (ii - 3) * (N_sample_Ts_jam + N_sample_tao_jam))];
    end
    Sig_jam = zeros(1, length(Sig_jam_matrix(1, :)));

    %% 合成干扰
    for ii = 2:R + 1
        Sig_jam = Sig_jam + Sig_jam_matrix(ii, :);
    end

    %% 调整长度，使 Sig_rec 和 Sig_jam 的长度一致
    if length(Sig_jam) > length(Sig_rec)
        Sig_jam = Sig_jam(1:length(Sig_rec));  % 裁剪 Sig_jam，使其与 Sig_rec 长度一致
    elseif length(Sig_jam) < length(Sig_rec)
        Sig_rec = Sig_rec(1:length(Sig_jam));  % 裁剪 Sig_rec，使其与 Sig_jam 长度一致
    end

end

    % 添加噪声
    s_rx = awgn(Sig_jam, SNR, 'measured');
    
%% 后续处理：脉冲压缩、STFT时频分析、二值化及形态学处理
%脉冲压缩（匹配滤波）
N_fft = 2 * N_sample;  % FFT 点数
Sig_ref = exp(1i * pi * Kr * (t) .^ 2);  % 参考信号
F_Sig_ref = fft(Sig_ref, N_fft);  % 参考信号的FFT

% 干扰信号的脉冲压缩
s_pc = fftshift(ifft(fft(s_rx, N_fft) .* conj(F_Sig_ref)));

%% STFT时频分析参数设置
window = 64;
noverlap = 60;
nfft = 512;

[S, f, t_stft] = spectrogram(s_pc, window, noverlap, nfft, fs);
TFD = abs(S);

%% 绘制干扰信号时域波形
figure;
plot(t * 1e6, real(Sig_jam));  % 绘制干扰信号的实部
xlabel('时间 (μs)');
ylabel('幅值');
title('干扰信号时域波形');
grid on;
% 绘制时频分布图
figure;
imagesc(t_stft*1e6, f/1e6, TFD);
axis xy;
xlabel('时间 (μs)');
ylabel('频率 (MHz)');
title('时频分布');
colorbar;

%% 二值化处理
gamma_w = 0.5;              % 设置二值化的阈值比例
max_s = max(TFD(:));
TFD_bin = TFD >= gamma_w * max_s;

% 绘制二值化时频图
figure;
imagesc(t_stft*1e6, f/1e6, TFD_bin);
axis xy;
xlabel('时间 (μs)');
ylabel('频率 (MHz)');
title('二值化时频图');
colorbar;
%% 形态学处理
se = ones(5);  % 定义结构元素
TFD_bin_processed = imerode(TFD_bin, se);  % 腐蚀操作
TFD_bin_processed = imdilate(TFD_bin_processed, se);  % 膨胀操作

% 绘制处理后的二值化时频图
figure;
imagesc(t_stft*1e6, f/1e6, TFD_bin_processed);
axis xy;
xlabel('时间 (μs)');
ylabel('频率 (MHz)');
title('形态学处理后的二值化时频图');
colorbar;
%% 投影分析连续块数目
% 时间轴投影（列求和）并二值化
time_proj = sum(TFD_bin_processed, 1);       % 沿时间轴投影
time_binary = time_proj > 0;                 % 投影二值化

% 频率轴投影（行求和）并二值化
freq_proj = sum(TFD_bin_processed, 2);       % 沿频率轴投影
freq_binary = freq_proj(:) > 0;              % 转为列向量并二值化

% 统计时间轴连续块数目
time_blocks = 0;
in_block = false;
for i = 1:length(time_binary)
    if time_binary(i)
        if ~in_block
            time_blocks = time_blocks + 1;
            in_block = true;
        end
    else
        in_block = false;
    end
end

% 统计频率轴连续块数目
freq_blocks = 0;
in_block = false;
for i = 1:length(freq_binary)
    if freq_binary(i)
        if ~in_block
            freq_blocks = freq_blocks + 1;
            in_block = true;
        end
    else
        in_block = false;
    end
end

% 输出分析结果
disp('===== 峰值块结构分析 =====');
disp(['时间轴连续块数：', num2str(time_blocks)]);
disp(['频率轴连续块数：', num2str(freq_blocks)]);
disp(['峰值块结构：', num2str(time_blocks), '×', num2str(freq_blocks)]);