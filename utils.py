import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# 写入 WAV 文件
def write_signal_to_wav(filename: str, data: np.ndarray, sampling_freq: float):
    wav.write(filename, int(sampling_freq), data)

# 从 WAV 文件读取信号
def from_wav_to_signal(filename: str) -> np.ndarray:
    return wav.read(filename)[1]

# 画出信号
def plot_signal(signal: np.ndarray, title: str = "", save: bool = False):
    plt.clf()
    plt.cla()
    plt.plot(signal)
    plt.title(title)
    plt.show()
    title = title.replace(" ", "_")
    # if save:
    #     plt.savefig(title + ".jpg", dpi=300)

# 调制/解调的配置
class Config:
    def __init__(self, sampling_freq: float, amplitude: float):
        self.sampling_freq = sampling_freq
        self.amplitude = amplitude

# 从 dB 转换到振幅
def from_db_to_amp(db: float, signal_amp: float) -> float:
    return signal_amp / (10 ** (db / 20))

# 在信号上加噪声
def add_noise(signal: np.ndarray, noise_amp: float) -> np.ndarray:
    noise = np.random.normal(0, noise_amp, len(signal))
    return signal + noise