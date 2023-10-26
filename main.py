import scipy.signal as sig
import scipy.io.wavfile as wav
import numpy as np
import ffmpeg
import os
import pyaudio

from matplotlib import pyplot as plt

# 写入 WAV 文件
def write_signal_to_wav(filename: str, data: np.ndarray, sampling_freq: float):
    wav.write(filename, sampling_freq, data)

# Problem 1:
# 1. 使用任意编程语言（如 C、C++、MATLAB、Python、Java、JavaScript 等），实现一个可以生成符合条件的声波信号的应用（本次作业所有小题均对应用 UI 无要求，可仅为命令行应用）（2 分）： 
# a) 要求生成的声波信号可以根据用户输入设置采样率、频率、初始相位和信号持续时间；
# b) 将生成的声波信号可以存储成 WAV 格式的音频文件，该文件可以通过系统标准播放器和 2 中实现的音频读取应用播放。
def generate_wave_signal(freq: float, phase: float, duration: float, sampling_freq: float) -> np.ndarray:
    t = np.arange(0, duration, 1 / sampling_freq)
    return np.sin(2 * np.pi * freq * t + phase)

def generate_wave_signal_to_file(filename: str, freq: float, phase: float, duration: float, sampling_freq: float):
    data = generate_wave_signal(freq, phase, duration, sampling_freq)
    write_signal_to_wav(filename, data, sampling_freq)


# Problem 2:
# 2. 使用任意编程语言（如 C、C++、MATLAB、Python、Java、JavaScript 等），实现一个可读取音频文件的函数（1 分）：
# a) 要求可以读取手机录制的音频文件；
# b) 利用 MATLAB 绘图函数（或其他编程语言绘图库）绘制信号波形图。
def from_music_to_wav(filename: str, output_filename: str):
    ffmpeg.input(filename).output(output_filename).run()

def from_wav_to_signal(filename: str) -> np.ndarray:
    return wav.read(filename)[1]

def plot_signal(signal: np.ndarray):
    plt.plot(signal)
    plt.show()


def plot_music(filename: str):
    # create a temp file to store the wav file
    tmp_file_name = "tmp.wav"
    # delete the file if it exists
    try:
        os.remove(tmp_file_name)
    except:
        pass
    from_music_to_wav(filename, tmp_file_name)
    signal = from_wav_to_signal(tmp_file_name)
    plot_signal(signal)
    # delete the temp file
    try:
        os.remove(tmp_file_name)
    except:
        pass

# 3. 使用任意编程语言（如 C、C++、MATLAB、Python、Java、JavaScript 等），调用电脑或手机麦克风，实现一个声波接收应用，要求能根据用户指定的采样频率和录音时长，将收到的声波存储为指定格式文件(WAV)（2 分）：
# a) 录音文件可通过系统标准播放器和 2 中实现的音频读取应用播放；
# b) 利用 MATLAB 绘图函数（或其他编程语言绘图库）绘制录音信号波形图。

def record_to_signal(sampling_freq: float, time: float) -> np.ndarray:
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=int(sampling_freq), input=True, frames_per_buffer=1024)
    frames = []
    for i in range(0, int(sampling_freq / 1024 * time)):
        data = stream.read(1024)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    return np.frombuffer(b''.join(frames), dtype=np.int16)


def record_to_wav(filename: str, sampling_freq: float, time: float):
    print(f"Recording for {time} second...", flush=True)
    data = record_to_signal(sampling_freq, time)
    print("End recording.", flush=True)
    write_signal_to_wav(filename, data, sampling_freq)


# 4. 理解基于幅度特征的信号调制与解调方法，编程实现基于开关键控（OOK）的信号调制与解调函数（4 分）：
# a) 调制函数的输入为 010011101100101，输出为调制后的声音信号，将声音信号保存为 WAV 格式文件；
# b) 使用 3 中实现的声波接收应用，接收 a)中产生的声音文件并保存为 WAV格式文件；
# c) 解调函数的输入为 b)中接收保存的声音文件，输出为解调后得到的二进制符号组合；
# d) 参数要求：采样频率 48kHz；信号频率 20kHz；振幅 1；每个符号调制长度为 25ms = 0.025s

# data is a list of 0 and 1
def modulate_signal(data: np.ndarray, sampling_freq: float, carrier_freq: float, amplitude: float, symbol_duration: float) -> np.ndarray:
    # padding a 1 at the beginning and the end of the data
    data = np.insert(data, 0, 1)
    data = np.append(data, 1)

    # then padding five 0's at the beginning and the end of the data to avoid popping sound
    data = np.insert(data, 0, np.zeros(5))
    data = np.append(data, np.zeros(5))

    t = np.arange(0, len(data) * symbol_duration, 1 / sampling_freq)
    # judge whether t is in the 0 period or 1 period as the time
    def fn(x: float):
        return data[int(x / symbol_duration)] == 1
    t_rect = np.vectorize(fn)(t)

    t_carrier = np.cos(2 * np.pi * carrier_freq * t)
    t_final = t_rect * t_carrier

    # return amplitude * t_final

    # add a band pass filter
    b, a = sig.iirfilter(1, [carrier_freq - 1000, carrier_freq + 1000], btype='bandpass', analog=False, ftype='butter', fs = sampling_freq)

    return amplitude * sig.lfilter(b, a, t_final)



def demodulate_signal(signal: np.ndarray, sampling_freq: float, carrier_freq: float, symbol_duration: float) -> np.ndarray:
    t = np.arange(0, len(signal)) / sampling_freq
    # t_carrier = np.cos(2 * np.pi * carrier_freq * t)
    # t_final = signal * t_carrier

    # create a band pass filter to filter the signal
    b, a = sig.iirfilter(1, [carrier_freq - 1000, carrier_freq + 1000], btype='bandpass', analog=False, ftype='butter', fs = sampling_freq)

    signal = sig.lfilter(b, a, signal)

    step = 10
    convolved_signal = sig.convolve(np.abs(signal), np.ones(step), mode='same')

    max_signal = np.max(convolved_signal)
    
    # find the start and the end of the signal
    start, end = 0, 0
    for i in range(len(signal)):
        if convolved_signal[i] > 0.5 * max_signal:
            start = i
            break
    for i in range(len(signal) - 1, -1, -1):
        if convolved_signal[i] > 0.5 * max_signal:
            end = i
            break

    print(f"start: {start} end: {end}")

    # cut the signal, left a 0.01 * symbol_duration margin
    start = max(int(start - 0.25 * symbol_duration * sampling_freq), 0)
    end = min(int(end + 0.25 * symbol_duration * sampling_freq), len(signal))
    signal = signal[start: end]

    # t_carrier = np.cos(2 * np.pi * carrier_freq * t[start: end])
    # signal = signal * t_carrier

    # normalize the signal
    signal = np.abs(signal)
    plot_signal(signal)    

    # a low pass filter
    b, a = sig.iirfilter(1, 4 / symbol_duration, btype='lowpass', analog=False, ftype='butter', fs = sampling_freq)
    t_filtered = 2 * sig.lfilter(b, a, signal)

    # normalize the signal
    t_filtered = t_filtered / np.max(t_filtered)

    plot_signal(t_filtered)

    # find the first place where the signal is more than 0.5 and the last place where the signal is more than 0.5
    # the data is between the two places
    # we want to justify the place that the signal is more than 0.5 for more than 0.5 * symbol_duration
    # every 0.01 * symbol_duration, we check whether the signal is more than 0.5


    # sampling the signal to vote for the data
    data_len = int(len(t_filtered) / (sampling_freq * symbol_duration))
    print("data_len: ", data_len)
    datas = np.zeros(data_len)
    # pick 100 points in each symbol duration
    for i in range(data_len):
        datas[i] = np.mean(t_filtered[int(i * sampling_freq * symbol_duration): int((i + 1) * sampling_freq * symbol_duration)])
    print("data: ", datas)
    plt.cla()
    x = np.arange(0, len(datas))
    plt.bar(x, datas)
    plt.show()
    # judge whether the data is 0 or 1
    def fn(x: float):
        return 1 if x > 0.5 else 0
    return np.vectorize(fn)(datas[1: -1])


def modulate_signal_to_file(filename: str, data: np.ndarray, sampling_freq: float, carrier_freq: float, amplitude: float, symbol_duration: float):
    modulated_signal = modulate_signal(data, sampling_freq, carrier_freq, amplitude, symbol_duration)
    plot_signal(modulated_signal)
    write_signal_to_wav(filename, modulated_signal, sampling_freq)


def demodulate_signal_from_file(filename: str, sampling_freq: float, carrier_freq: float, symbol_duration: float):
    signal = from_wav_to_signal(filename)
    plot_signal(signal)
    demodulated_signal = demodulate_signal(signal, sampling_freq, carrier_freq, symbol_duration)
    print(demodulated_signal)


if __name__ == "__main__":
    # 生成信号
    # generate_wave_signal_to_file("test.wav", 440, 0, 2, 44100)

    # 读取音频文件
    # plot_music("record.wav")


    # 录音

    # 调制
    data = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 0])
    sampling_freq = 48000
    carrier_freq = 20000
    amplitude = 1
    symbol_duration = 0.025

    # modulate_signal_to_file("modulated_signal.wav", data, sampling_freq, carrier_freq, amplitude, symbol_duration)

    # 解调
    record_to_wav("recorded_signal.wav", sampling_freq, 2)
    demodulate_signal_from_file("recorded_signal.wav", sampling_freq, carrier_freq, symbol_duration)





