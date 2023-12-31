# IOT Homework

## 作业1: 调制解调

### 依赖安装

需要依赖 ffmpeg 进行转码，请将 ffmpeg 的二进制可执行文件放置于 PATH 中，否则输入文件只能为 `.wav` 格式。

其他依赖可以通过 `pip install -r requirements.txt` 安装。

作业代码位于 `hw1.py`。

### 使用方法提示

`python ./hw1.py --help`

得到：

```
python .\hw1.py --help
usage: hw1.py [-h] [--sampling_freq SAMPLING_FREQ] [--carrier_freq CARRIER_FREQ] [--amplitude AMPLITUDE]
               [--symbol_duration SYMBOL_DURATION] [--duration DURATION] [--data DATA [DATA ...]]
               function filename

IOT Homework for THSS.

positional arguments:
  function              the function to run
  filename              the filename to use

options:
  -h, --help            show this help message and exit
  --sampling_freq SAMPLING_FREQ
                        the sampling frequency (in Hz)
  --carrier_freq CARRIER_FREQ
                        the carrier frequency (in Hz)
  --amplitude AMPLITUDE
                        the amplitude
  --symbol_duration SYMBOL_DURATION
                        the symbol duration in (de)modulating(in ms)
  --duration DURATION   the recording/generating duration(in s)
  --data DATA [DATA ...]
                        the data to modulate
```

通过命令行输入参数。

具体如何使用请见 report/report1.md 。

## 作业2

### 依赖安装

仅需 `pip install -r requirements.txt` 安装依赖。

### 运行代码

`python ./hw2.py`

具体报告请见 report/report2.md 。

## 作业3

### 依赖安装

仅需 `pip install -r requirements.txt` 安装依赖。

### 运行代码

`python ./hw3.py`

具体报告请见 report/report3.md 。

## 作业4

### 依赖安装

仅需 `pip install -r requirements.txt` 安装依赖。

### 运行代码

`python ./hw4.py`

具体报告请见 report/report4.md 。