
import numpy as np
from utils import Config, write_signal_to_wav, from_wav_to_signal, plot_signal
import scipy.signal as sig

# Problem 1:
# 理解基于脉冲间隔的信号调制与解调方法，编程实现基于脉冲间隔的信号调制函数与解调函数:（2分）
# a)调制函数的输入为010011101100101，输出为调制后的声音信号，将声音信号保存成WAV格式文件。
# b)解调函数的输入为a)中产生的声音文件，输出为解调后得到的二进制符号组合。
# c)参数要求：采样频率48kHz；脉冲信号频率20kHz；振幅1；起始相位0；脉冲持续时间10ms；脉冲间隔：20ms（比特0）、30ms（比特1）。
# 
# 注：此处“脉冲间隔”指相邻比特信号开始时间的时间差。

class PIMConfig(Config):
    def __init__(self, sampling_freq: float, pulse_freq: float, amplitude: float, phase: float, pulse_duration: float, pulse_interval_0: float, pulse_interval_1: float):
        super().__init__(sampling_freq, amplitude)
        self.pulse_freq = pulse_freq
        self.phase = phase
        self.pulse_duration = pulse_duration
        self.pulse_interval_0 = pulse_interval_0
        self.pulse_interval_1 = pulse_interval_1

def PIM_modulation(data: np.ndarray, PIM_config: PIMConfig, plot: bool = False) -> np.ndarray:

    pulse_signal = np.arange(0, PIM_config.pulse_duration, 1 / PIM_config.sampling_freq)
    pulse_signal = PIM_config.amplitude * np.sin(2 * np.pi * PIM_config.pulse_freq * pulse_signal + PIM_config.phase)
    signal_0 = np.concatenate((pulse_signal, np.zeros(int((PIM_config.pulse_interval_0 - PIM_config.pulse_duration) * PIM_config.sampling_freq))))
    signal_1 = np.concatenate((pulse_signal, np.zeros(int((PIM_config.pulse_interval_1 - PIM_config.pulse_duration) * PIM_config.sampling_freq))))
    signal = np.concatenate([signal_0 if bit == 0 else signal_1 for bit in data])
    # padding: 前面加上 0.03s 的静音，后面加上一个脉冲
    padding_time = 0.03
    padding_start = np.zeros(int(padding_time * PIM_config.sampling_freq))
    padding_end = pulse_signal
    signal = np.concatenate((padding_start, signal, padding_end))
    if plot:
        plot_signal(signal, "PIM Modulated Signal")
    return signal

def PIM_demodulation(signal: np.ndarray, PIM_config: PIMConfig, plot: bool = False) -> np.ndarray:
    # 振幅的阈值
    threshold_amp = PIM_config.amplitude / 2 
    # 信号长度的阈值
    threshold_interval = (PIM_config.pulse_interval_0 + PIM_config.pulse_interval_1) / 2
    
    # 带通滤波
    b, a = sig.iirfilter(
        3,
        [PIM_config.pulse_freq - 100, PIM_config.pulse_freq + 100],
        btype="bandpass",
        analog=False,
        ftype="butter",
        fs=PIM_config.sampling_freq,
    )
    signal = sig.lfilter(b, a, signal)
    if plot:
        plot_signal(signal, "PIM Flitered Signal")

    
    amp = np.abs(sig.hilbert(signal))
    if plot:
        plot_signal(amp, "PIM Demodulated Signal")

    result = []
    start = 0

    for i in range(1, len(amp)):
        if amp[i] >= threshold_amp and (i == 1 or amp[i - 1] < threshold_amp):
            if start == 0:
                start = i
                continue
            # print(i - start)
            # print(threshold_interval * PIM_config.sampling_freq)
            if i - start > threshold_interval * PIM_config.sampling_freq:
                result.append(1)
            else:
                result.append(0)
            start = i
    
    result_signal = np.array(result)

    return result_signal                
    

# 2.理解基于相位特征的信号调制与解调方法，编程实现二元相移键控（BPSK）信号调制函数与解调函数：（3分）
# a)调制函数的输入为0100111011001010，输出为调制后的声音信号，将声音信号保存成WAV格式文件。
# b)解调函数的输入为a)中产生的声音文件，输出为解调后得到的二进制符号组合。
# c)参数要求：采样频率48kHz；信号频率20kHz；振幅1；调制符号长度25ms。

class BPSKConfig(Config):
    def __init__(self, sampling_freq: float, signal_freq: float, amplitude: float, symbol_duration: float):
        super().__init__(sampling_freq, amplitude)
        self.signal_freq = signal_freq
        self.symbol_duration = symbol_duration

def BPSK_modulation(data: np.ndarray, BPSK_config: BPSKConfig, plot: bool = False) -> np.ndarray:
    signal = np.arange(0, BPSK_config.symbol_duration, 1 / BPSK_config.sampling_freq)
    signal = BPSK_config.amplitude * np.cos(2 * np.pi * BPSK_config.signal_freq * signal)
    signal_0 = signal # \phi = 0
    signal_1 = -signal # \phi = \pi
    signal = np.concatenate([signal_0 if bit == 0 else signal_1 for bit in data])
    if plot:
        plot_signal(signal, "BPSK modulated signal")
    return signal

def BPSK_demodulation(signal: np.ndarray, BPSK_config: BPSKConfig, plot: bool = False) -> np.ndarray:

    # 相干解调
    time_step = 1 / BPSK_config.sampling_freq
    carrier_signal = np.arange(0, len(signal), 1) * time_step
    carrier_signal = np.cos(2 * np.pi * BPSK_config.signal_freq * carrier_signal)  # \phi = 0
    signal = signal * carrier_signal / BPSK_config.amplitude
    # 低通滤波
    b, a = sig.iirfilter(
        1,
        4 / BPSK_config.symbol_duration,
        btype="lowpass",
        analog=False,
        ftype="butter",
        fs=BPSK_config.sampling_freq,
    )
    signal = sig.lfilter(b, a, signal)

    # padding 填充一个符号长度的静音
    padding_time = BPSK_config.symbol_duration
    padding_start = np.zeros(int(padding_time * BPSK_config.sampling_freq))
    padding_end = np.zeros(int(padding_time * BPSK_config.sampling_freq))
    signal = np.concatenate((padding_start, signal, padding_end))

    # 获取 start 和 end

    start = 0
    end = 0

    for i in range(len(signal)):
        if abs(signal[i]) > 0.25 and abs(signal[i - 1]) <= 0.25:
            start = i
            break
    
    for i in range(len(signal) - 1, 0, -1):
        if abs(signal[i]) > 0.25 and abs(signal[i + 1]) <= 0.25:
            end = i
            break
    
    data_length = round((end - start) / BPSK_config.sampling_freq / BPSK_config.symbol_duration)
    
    signal = signal[start:end]
    if plot:
        plot_signal(signal, "BPSK demodulated signal")

    # 采样取平均
    datas = np.zeros(data_length)
    for i in range(data_length):
        symbol_start = int(i * BPSK_config.symbol_duration * BPSK_config.sampling_freq)
        symbol_end = int((i + 1) * BPSK_config.symbol_duration * BPSK_config.sampling_freq)
        symbol_start = max(symbol_start, 0)
        symbol_end = min(symbol_end, len(signal))
        datas[i] = np.mean(signal[symbol_start:symbol_end])

    # 二值化
    result = np.zeros(data_length)
    for i in range(data_length):
        if datas[i] > 0:
            result[i] = 0
        else:
            result[i] = 1
    return result



# 3.验证调制、解调算法在噪声环境下的性能：（4分）
# a)编写程序，在1，2小题产生的调制信号中加入不同程度的加性高斯白噪声（AWGN）：调整白噪声方差，模拟产生信噪比为20dB、10dB、5dB、0dB的信号。请分别测量脉冲调制和BPSK调制在上述信噪比下的传输成功率（正确传输比特数/总传输比特数），以图或表的形式展示测量结果。
# b)BPSK中，调制符号长度对解码正确率有一定影响。修改BPSK代码中的调制符号长度，将其调整为原长度的2倍、4倍、8倍、16倍，在信噪比为10dB时，分别测量上述各符号长度对应的传输成功率。

def from_db_to_amp(db: float, signal_amp: float) -> float:
    return signal_amp / (10 ** (db / 20))

def add_noise(signal: np.ndarray, noise_amp: float) -> np.ndarray:
    noise = np.random.normal(0, noise_amp, len(signal))
    return signal + noise


def test_PIM(data: np.ndarray, PIM_config: PIMConfig) -> float:
    signal = PIM_modulation(data, PIM_config, True)
    write_signal_to_wav("PIM.wav", signal, PIM_config.sampling_freq)
    read_signal = from_wav_to_signal("PIM.wav")
    result = PIM_demodulation(read_signal, PIM_config, True)
    print(f'PIM, data={data}, result={result}, correct rate={np.mean(result == data)}')
    return np.mean(result == data)

def test_BPSK(data: np.ndarray, BPSK_config: BPSKConfig) -> float:
    signal = BPSK_modulation(data, BPSK_config, True)
    write_signal_to_wav("BPSK.wav", signal, BPSK_config.sampling_freq)
    read_signal = from_wav_to_signal("BPSK.wav")
    result = BPSK_demodulation(read_signal, BPSK_config, True)
    print(f'BPSK, data={data}, result={result}, correct rate={np.mean(result == data)}')
    return np.mean(result == data)

def test_PIM_noise(data: np.ndarray, PIM_config: PIMConfig, noise_db: float) -> float:
    signal = PIM_modulation(data, PIM_config)
    signal = add_noise(signal, from_db_to_amp(noise_db, PIM_config.amplitude))
    plot_signal(signal, f"PIM Modulated Signal with Noise {noise_db} DB", True)
    result = PIM_demodulation(signal, PIM_config, True)
    return np.mean(result == data)

def test_BPSK_noise(data: np.ndarray, BPSK_config: BPSKConfig, noise_db: float) -> float:
    signal = BPSK_modulation(data, BPSK_config)
    signal = add_noise(signal, from_db_to_amp(noise_db, BPSK_config.amplitude))
    plot_signal(signal, f"BPSK Modulated Signal with Noise {noise_db} DB", True)
    result = BPSK_demodulation(signal, BPSK_config, True)
    return np.mean(result == data)

if __name__ == "__main__":
    # PIM
    pim_data = np.array([0,1,0,0,1,1,1,0,1,1,0,0,1,0,1])
    pim_config = PIMConfig(48000, 20000, 1, 0, 0.01, 0.02, 0.03) 
    # test_PIM(pim_data, pim_config)

    # BPSK
    bpsk_data = np.array([0,1,0,0,1,1,1,0,1,1,0,0,1,0,1,0])
    bpsk_config = BPSKConfig(48000, 20000, 1, 0.025)
    # test_BPSK(bpsk_data, bpsk_config)

    # Noise
    noise_dbs = [20, 10, 5, 0]
    noise_correct_rates = []
    
    #   PIM
    for noise_db in noise_dbs:
        noise_correct_rates.append(test_PIM_noise(pim_data, pim_config, noise_db))
    noise_str = [f"\t{noise_dbs[i]}dB: {noise_correct_rates[i]}\n" for i in range(len(noise_dbs))]
    print(f"PIM correct_rate:\n{''.join(noise_str)}")

    #   BPSK
    for noise_db in noise_dbs:
        noise_correct_rates.append(test_BPSK_noise(bpsk_data, bpsk_config, noise_db))
    noise_str = [f"\t{noise_dbs[i]}dB: {noise_correct_rates[i]}\n" for i in range(len(noise_dbs))]
    print(f"BPSK correct_rate:\n{''.join(noise_str)}")

    # Length of BPSK symbol
    times = [2, 4, 8, 16]
    length_correct_rates = []

    for time in times:
        bpsk_config.symbol_duration = 0.025 * time
        length_correct_rates.append(test_BPSK_noise(bpsk_data, bpsk_config, 10))
    length_str = [f"\tTime {times[i]}: {length_correct_rates[i]}\n" for i in range(len(times))]
    print(f"BPSK correct_rate:\n{''.join(length_str)}")
    


