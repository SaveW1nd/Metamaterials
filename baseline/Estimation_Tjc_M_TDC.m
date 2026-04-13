%% 间歇采样转发干扰参数估计完整代码
clear; clc; close all;

%% 参数设置
Tp = 20e-6;                 % 雷达信号脉宽 24us
B = 100e6;                  % 雷达信号带宽 100MHz
Kr = B/Tp;                  % 调频斜率
fs = 2*B;                   % 采样率 200MHz
Ts = 1/fs;                  % 采样间隔
N_sample = ceil(Tp/Ts);     % 总采样点数

% ISRJ干扰参数
tao_jam = 1e-6;             % 切片宽度 1us
M = 3;                      % 转发次数
Ts_jam = (M+1)*tao_jam;     % 采样周期 (4us)
N_tao = ceil(tao_jam/Ts);   % 切片采样点数
N_Tsjam = ceil(Ts_jam/Ts);  % 周期采样点数

%% 生成雷达发射信号
t = (0:N_sample-1)*Ts;      % 时间序列
Sig_tx = exp(1i*pi*Kr*t.^2); % LFM发射信号

%% 生成间歇采样转发干扰
% 生成采样脉冲模板
Sig_pulse = zeros(1, N_sample);
N_slices = floor(Tp/Ts_jam); % 切片数量
for k = 1:N_slices
    start_idx = (k-1)*N_Tsjam + 1;
    end_idx = start_idx + N_tao - 1;
    if end_idx > N_sample
        end_idx = N_sample;
    end
    Sig_pulse(start_idx:end_idx) = 1;
end

% 生成基础干扰信号
Sig_jam_base = Sig_tx .* Sig_pulse;

% 调整干扰功率
JSR_dB = 10;                % 干信比10dB
P_sig = mean(abs(Sig_tx).^2);
P_jam = mean(abs(Sig_jam_base).^2);
Sig_jam_base = Sig_jam_base * sqrt(P_sig*10^(JSR_dB/10)/P_jam);

% 生成重复转发干扰
Sig_jam = zeros(1, N_sample);
for m = 1:M
    delay = m*N_tao;        % 时延采样点数
    start_idx = delay + 1;
    end_idx = min(start_idx + N_sample - 1, N_sample);
    Sig_jam(start_idx:end_idx) = Sig_jam(start_idx:end_idx) + ...
                                 Sig_jam_base(1:end_idx-start_idx+1);
end

%% TDC处理流程
% 解线调处理
Sig_dechirp = Sig_jam .* conj(Sig_tx);

% 频域反卷积
F_tx = fft(Sig_tx);         % 发射信号频谱
F_jam = fft(Sig_dechirp);   % 干扰信号频谱
epsilon = 1e-6;             % 正则化系数
F_tdc = F_jam ./ (F_tx + epsilon); % 频域反卷积

% 时域反卷积结果
Sig_tdc = ifft(F_tdc);

%% 冲激脉冲检测与参数估计
% 峰值检测参数
min_peak_dist = floor(N_tao/2); % 最小脉冲间隔
threshold = 0.3 * max(abs(Sig_tdc)); % 动态阈值

% 检测峰值
[peaks, locs] = findpeaks(abs(Sig_tdc),...
    'MinPeakHeight', threshold,...
    'MinPeakDistance', min_peak_dist);

% 脉冲对匹配
pulse_pairs = [];
for i = 1:length(locs)-1
    time_diff = (locs(i+1)-locs(i)) * Ts;
    if time_diff >= tao_jam*0.8 && time_diff <= tao_jam*1.2
        pulse_pairs = [pulse_pairs; locs(i), locs(i+1)];
    end
end

% 参数估计
Q_est = size(pulse_pairs,1); % 转发次数估计
if Q_est > 0
    widths = (pulse_pairs(:,2)-pulse_pairs(:,1)) * Ts;
    tao_est = mean(widths);  % 切片宽度估计
else
    tao_est = 0;
end

%% 结果展示
figure('Position',[100 100 800 600])
subplot(211)
plot(t*1e6, real(Sig_jam))
% title('原始干扰信号时域波形')
xlabel('Time(μs)'), ylabel('Amplitude')
hold on
plot(t*1e6, abs(Sig_tdc))
hold on
plot(t(locs)*1e6, peaks, 'ro')
% title('TDC处理后冲激脉冲检测')
% xlabel('时间(\mus)'), ylabel('幅度')
legend('signal','detection pulse')

subplot(212)
stem(pulse_pairs(:,1)*Ts*1e6, ones(size(pulse_pairs,1)), 'filled')
hold on
stem(pulse_pairs(:,2)*Ts*1e6, ones(size(pulse_pairs,1)), 'filled')
title('脉冲位置标记')
xlabel('时间(\mus)'), ylabel('标记')
xlim([0 Tp*1e6])

fprintf('===== 估计结果 =====\n');
fprintf('真实切片宽度: %.2f μs\n', tao_jam*1e6);
fprintf('估计切片宽度: %.2f μs\n', tao_est*1e6);
fprintf('真实转发次数: %d\n', M);
fprintf('估计转发次数: %d\n', Q_est);