import numpy as np
from utils import Config, write_signal_to_wav, from_wav_to_signal, plot_signal, add_noise, from_db_to_amp
import scipy.signal as sig
import scipy.fft as fft
from matplotlib import pyplot as plt

from typing import Callable, List

# 1. 理解基于相位特征的信号调制与解调方法，编程实现正交相移键控（QPSK）信号调制函数与解调函数：（2 分） 
#   a) 调制函数的输入为 0100111011001010 ，输出为调制后的声音信号，将声音信号保存成 WAV 格式文件。注：将原始数据拆分为 IQ 两路进行调制。
#   b) 解调函数的输入为 a)中产生的声音文件，输出为解调后得到的二进制符号组合。
#   c) 参数要求：采样频率48kHz；信号频率20kHz；振幅1；调制符号长度25ms。

class QPSKConfig(Config):
    def __init__(self, sampling_freq: float, signal_freq: float, amplitude: float, signal_duration: float):
        super().__init__(sampling_freq, amplitude)
        self.signal_freq = signal_freq
        self.signal_duration = signal_duration

def QPSK_modulation(data: np.ndarray, QPSK_config: QPSKConfig, plot: bool = False) -> np.ndarray:
    if len(data) % 2 != 0:
        data = np.append(data, [0])
    sigI = np.arange(0, QPSK_config.signal_duration, 1 / QPSK_config.sampling_freq)
    sigI = QPSK_config.amplitude * np.sin(2 * np.pi * QPSK_config.signal_freq * sigI)
    sigQ = np.arange(0, QPSK_config.signal_duration, 1 / QPSK_config.sampling_freq)
    sigQ = QPSK_config.amplitude * np.cos(2 * np.pi * QPSK_config.signal_freq * sigQ)

    signal_length = int(len(data) / 2 * len(sigI))

    signal = np.zeros(signal_length)

    for i in range(0, len(data), 2):
        fI = (1 - 2 * data[i]) * sigI
        fQ = (1 - 2 * data[i + 1]) * sigQ
        signal[i // 2 * len(sigI): (i // 2 + 1) * len(sigI)] = fI + fQ
    
    signal = signal * np.sqrt(2) / 2

    if plot:
        plot_signal(signal, QPSK_config.sampling_freq, QPSK_config.signal_freq, QPSK_config.signal_duration)

    return signal

def QPSK_demodulation(signal: np.ndarray, QPSK_config: QPSKConfig, plot: bool = False) -> np.ndarray:
    sigI = np.arange(0, QPSK_config.signal_duration, 1 / QPSK_config.sampling_freq)
    sigI = QPSK_config.amplitude * np.sin(2 * np.pi * QPSK_config.signal_freq * sigI)
    sigQ = np.arange(0, QPSK_config.signal_duration, 1 / QPSK_config.sampling_freq)
    sigQ = QPSK_config.amplitude * np.cos(2 * np.pi * QPSK_config.signal_freq * sigQ)

    sigL = len(sigI)
    # build a signal matrix, each row is a signal
    sigMat = np.zeros((4, sigL))
    sigMat[0] = sigI + sigQ
    sigMat[1] = sigI - sigQ
    sigMat[2] = -sigI + sigQ
    sigMat[3] = -sigI - sigQ


    data = np.zeros(round(len(signal) * 2 / sigL))
    data_len = len(data)
    for i in range(0, data_len, 2):
        seg = signal[i // 2 * sigL: (i // 2 + 1) * sigL]
        maxI = np.argmax(np.dot(sigMat, seg)) + 1
        data[i] = maxI > 2
        data[i + 1] = maxI % 2 == 0
     
    if plot:
        plot_signal(data, "QPSK demodulated")
    
    return data

def test_QPSK(data: np.ndarray, QPSK_config: QPSKConfig) -> float:
    signal = QPSK_modulation(data, QPSK_config)
    write_signal_to_wav("QPSK.wav", signal, QPSK_config.sampling_freq)
    read_signal = from_wav_to_signal("QPSK.wav")
    demodulated = QPSK_demodulation(read_signal, QPSK_config)
    print(f"QPSK, data={data}, demodulated={demodulated}, correct rate = {np.mean(data == demodulated)}")
    return np.mean(data == demodulated)

def test_QPSK_noise(data: np.ndarray, QPSK_config: QPSKConfig, noise_db: float) -> float:
    signal = QPSK_modulation(data, QPSK_config)
    signal = add_noise(signal, from_db_to_amp(noise_db, QPSK_config.amplitude))
    write_signal_to_wav("QPSK.wav", signal, QPSK_config.sampling_freq)
    read_signal = from_wav_to_signal("QPSK.wav")
    demodulated = QPSK_demodulation(read_signal, QPSK_config)
    print(f"QPSK, data={data}, demodulated={demodulated}, correct rate = {np.mean(data == demodulated)}")
    return np.mean(data == demodulated)


# 3. 当 N = 16, 64 和 1024 时，分别编程计算下列长度为 N 的序列的 N 点 DFT，并根据计算结果，绘制频谱图：（3 分）

# (1)  y[n] = 1, 0 <= n < N; 0, 其他
# (2)  y[n] = 1 - |n|/N, 0 <= n < N; 0, 其他
# (3)  y[n] = sin(2πn/N), 0 <= n < N; 0, 其他

def DFT(x: np.ndarray, r: int = 1) -> np.ndarray:
    fft_x = fft.fft(x, r * len(x))
    # 将负频率放到左侧
    fft_x = np.fft.fftshift(fft_x)
    return fft_x

def DFT_N(N: int, f: Callable[[int], np.ndarray], r: int = 1):
    x = f(N)
    fft_x = DFT(x, r)
    return fft_x

def get_x_axis(N: int, r: int = 1) -> np.ndarray:
    return (np.arange(0, r * N) / (r * N)) - 0.5    

def f1(N: int) -> np.ndarray:
    x = np.zeros(N)
    x[:N] = 1
    return x

def f2(N: int) -> np.ndarray:
    x = np.zeros(N)
    x[:N] = 1 - np.abs(np.arange(N) - N / 2) / N
    return x

def f3(N: int) -> np.ndarray:
    x = np.zeros(N)
    x[:N] = np.sin(2 * np.pi * np.arange(N) / N)
    return x

def test_DFT(N: int, funcs: List[Callable[[int], np.ndarray]] = [f1, f2, f3]):
    r = 16
    # draw a pic with several subplots
    fig, axes = plt.subplots(len(funcs), 1, figsize=(10, 10))
    for i in range(len(funcs)):
        x = get_x_axis(N, r)
        y = DFT_N(N, funcs[i], r)
        print(x.shape, y.shape)
        axes[i].scatter(x, np.abs(y))
        axes[i].set_title(f"N={N}, f{i+1}, r={r}")
    fig.tight_layout()
    fig.savefig(f"N={N}.jpg", dpi=300)




if __name__ == '__main__':
    data = np.array([0,1,0,0,1,1,1,0,1,1,0,0,1,0,1,0])
    QPSK_config = QPSKConfig(48000, 20000, 1, 0.025)
    test_QPSK(data, QPSK_config)

    noise_dbs = [20, 10, 5, 0, -30, -50]
    noise_correct_rates = []
    for noise_db in noise_dbs:
        result = test_QPSK_noise(data, QPSK_config, noise_db)
        noise_correct_rates.append(result)
    noise_str = [f"\t{noise_dbs[i]}dB: {noise_correct_rates[i]}\n" for i in range(len(noise_dbs))]
    print(f"QPSK correct_rate:\n{''.join(noise_str)}")


    # length of QPSK symbol
    times = [2, 4, 8, 16]
    length_correct_rates = []
    for time in times:
        QPSK_config.signal_duration = 0.025 * time
        result = test_QPSK_noise(data, QPSK_config, 10)
        length_correct_rates.append(result)
    length_str = [f"\t{times[i]}: {length_correct_rates[i]}\n" for i in range(len(times))]
    print(f"QPSK correct_rate:\n{''.join(length_str)}")
    
    test_DFT(16)
    test_DFT(64)
    test_DFT(1024)

