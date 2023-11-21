# 1. 编写程序，对声音信号文件（res.wav）进行分析：（4 分） 
# a) 分析信号频率组成：解析 WAV 文件提取音频信号。对提取到的信号做离散傅里叶变换，绘制信号的频谱图。
# b) 补零：再采用信号尾部补 0，使信号长度延展为原序列的 10 倍。对补零后信号再进行离散傅里叶变换，绘制信号的频谱图，分析补零对信号频谱的影响。
# c) 时频分析：对信号做短时傅里叶变换，绘制信号的时频图，即信号频率随时间变换的情况。改变短时傅里叶变换的窗口长度，分析窗口长度对变换结果的影响。
# 注：以上分析步骤允许使用现有函数（如 FFT，spectrogram 等）。

from utils import from_wav_to_signal, plot_freq, plot_spectrogram
import numpy as np
from scipy import fft


# a)
signal = from_wav_to_signal("res.wav")
plot_freq(signal, 48000, 1, "res.wav fft", True)

# b)
signal = from_wav_to_signal("res.wav")
plot_freq(signal, 48000, 10, "res.wav fft with zeros", True)

# c)
signal = from_wav_to_signal("res.wav")
plot_spectrogram(signal, 48000, 16, "res.wav spectrogram windows=16", True)
plot_spectrogram(signal, 48000, 64, "res.wav spectrogram windows=64", True)
plot_spectrogram(signal, 48000, 256, "res.wav spectrogram windows=256", True)


# 2. 利用时域和频域信号的对偶性质，用基于离散傅里叶变换的方法，编写程序计算两个长度为 N 的序列对的循环卷积：（3 分） 
# a) 要求可以根据用户输入的序列对计算循环卷积。
# b) 用该程序求下列序列对的循环卷积：
# i. g[n] = {3, 2, -2, 1, 0, 1}, h[n] = {-5, -1, 3, -2, 4, 4}
# ii. x[n] = cos(\pi n / 2), y[n] = 3^n, 0 <= n <= 4

# a)
def cyclic_convolution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # x and y must have the same length
    assert len(x) == len(y)
    N = len(x)
    # x and y are both in time domain
    X = fft.fft(x)
    Y = fft.fft(y)
    # X and Y are both in frequency domain
    Z = X * Y
    # Z is in frequency domain
    z = fft.ifft(Z)
    # z is in time domain
    # z is the cyclic convolution of x and y
    result = np.real(z)
    # keep two float point
    print(f'input x: {np.around(x, 2)}')
    print(f'input y: {np.around(y, 2)}')
    print(f'output z: {np.around(result, 2)}')
    return result

# b)
g = np.array([3, 2, -2, 1, 0, 1])
h = np.array([-5, -1, 3, -2, 4, 4])

cyclic_convolution(g, h)

x = np.cos(np.pi * np.arange(5) / 2)
y = 3 ** np.arange(5)
cyclic_convolution(x, y)

# from user input
print("input x and y, split with space.")
x = np.array(input("input x: ").split(), dtype=np.float32)
y = np.array(input("input y: ").split(), dtype=np.float32)
cyclic_convolution(x, y)