% Single-file ISRJ generation and plotting script.

clear; clc; close all;

bandwidth = 40e6;  % Coarse estimate from Fig. 2; not explicitly listed in the paper.
pulse_width = 40e-6;
fc = 0e6;
fs = 100e6;
pri = 40e-6;
N_samples = 4000;
target_delay = 0;
slice_width = 2e-6;
metasurface_floor = 0.5;
snr_db = 0;
jnr_db = 5;

forwarding_width = slice_width;
forwarding_delay = slice_width;
jammer_delay = target_delay + forwarding_delay;

num_time = N_samples;
num_pulse = round(pulse_width * fs);
t = (0:num_time - 1).' / fs;
tp = (0:num_pulse - 1).' / fs;

chirp_rate = bandwidth / pulse_width;
lfm_pulse = exp(1j * 2 * pi * (fc * tp + 0.5 * chirp_rate * tp .^ 2));

lfm_echo = complex(zeros(num_time, 1));
target_start = round(target_delay * fs) + 1;
target_stop = min(num_time, target_start + num_pulse - 1);
if target_start <= num_time
    target_count = target_stop - target_start + 1;
    lfm_echo(target_start:target_stop) = lfm_pulse(1:target_count);
end

jammer_echo = complex(zeros(num_time, 1));
jammer_start = round(jammer_delay * fs) + 1;
jammer_stop = min(num_time, jammer_start + num_pulse - 1);
if jammer_start <= num_time
    jammer_count = jammer_stop - jammer_start + 1;
    jammer_echo(jammer_start:jammer_stop) = lfm_pulse(1:jammer_count);
end

sampling_interval = slice_width + forwarding_width;
relative_t = t - jammer_delay;
gate = double(relative_t >= 0 & mod(relative_t, sampling_interval) < forwarding_width);
mask = metasurface_floor + (1 - metasurface_floor) * gate;
isrj = jammer_echo .* mask;

lfm_power = mean(abs(lfm_echo) .^ 2);
isrj_power = mean(abs(isrj) .^ 2);
jammer_scale = sqrt((lfm_power * 10 ^ (jnr_db / 10)) / max(isrj_power, eps));
isrj = jammer_scale * isrj;

noise_power = lfm_power / (10 ^ (snr_db / 10));
noise_sigma = sqrt(noise_power / 2);
noise = noise_sigma * (randn(num_time, 1) + 1j * randn(num_time, 1));
received = lfm_echo + isrj + noise;

win = 64;
noverlap = win - 1;
nfft = 1024;
[spec_lfm, f_lfm_tf, t_lfm_tf] = spectrogram(lfm_echo, hamming(win), noverlap, nfft, fs, 'centered');
[spec_isrj, f_isrj_tf, t_isrj_tf] = spectrogram(isrj, hamming(win), noverlap, nfft, fs, 'centered');

outdir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

fig = figure('Visible', 'off');
plot(t * 1e6, real(received), 'LineWidth', 1.0);
grid on;
xlabel('Time (\mus)');
ylabel('Real Part');
title('Received Signal Time Domain');
received_time_file = fullfile(outdir, 'received_time_domain.png');
exportgraphics(fig, received_time_file);
close(fig);

fig = figure('Visible', 'off');
plot(t * 1e6, real(lfm_echo), 'LineWidth', 1.0);
grid on;
xlabel('Time (\mus)');
ylabel('Real Part');
title('LFM Time Domain');
lfm_time_file = fullfile(outdir, 'lfm_time_domain.png');
exportgraphics(fig, lfm_time_file);
close(fig);

fig = figure('Visible', 'off');
plot(t * 1e6, real(isrj), 'LineWidth', 1.0);
grid on;
xlabel('Time (\mus)');
ylabel('Real Part');
title('ISRJ Time Domain');
isrj_time_file = fullfile(outdir, 'isrj_time_domain.png');
exportgraphics(fig, isrj_time_file);
close(fig);

fig = figure('Visible', 'off');
imagesc(t_lfm_tf * 1e6, f_lfm_tf / 1e6, abs(spec_lfm));
axis xy;
xlabel('Time (\mus)');
ylabel('Frequency (MHz)');
title('LFM Spectrogram');
colorbar;
lfm_output_file = fullfile(outdir, 'lfm_spectrogram.png');
exportgraphics(fig, lfm_output_file);
close(fig);

fig = figure('Visible', 'off');
imagesc(t_isrj_tf * 1e6, f_isrj_tf / 1e6, abs(spec_isrj));
axis xy;
xlabel('Time (\mus)');
ylabel('Frequency (MHz)');
title(sprintf('ISRJ Spectrogram (x = %.2f, SNR = %.1f dB, JNR = %.1f dB)', ...
    metasurface_floor, snr_db, jnr_db));
colorbar;
isrj_output_file = fullfile(outdir, 'isrj_spectrogram.png');
exportgraphics(fig, isrj_output_file);
close(fig);

fprintf('Saved %s\n', received_time_file);
fprintf('Saved %s\n', lfm_time_file);
fprintf('Saved %s\n', isrj_time_file);
fprintf('Saved %s\n', lfm_output_file);
fprintf('Saved %s\n', isrj_output_file);
