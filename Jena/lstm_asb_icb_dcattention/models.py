import torch
import torch.nn as nn
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import torch
import torch.nn as nn
import numpy as np
import torch
from torch import nn
from torch.nn import init

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim=(-d))
        r = torch.stack((t.real, t.imag), -1)
        return r


    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:, :, 0], x[:, :, 1]), dim=(-d))
        return t.real

"""
这段代码实现了一个名为 `dct_channel_block` 的 PyTorch 模块，该模块包含了离散余弦变换（DCT）和通道注意力机制的功能。

首先，定义了一个函数 `dct`，用于对输入的时间序列数据进行离散余弦变换。在函数内部，首先将输入的数据进行处理，然后利用快速傅里叶变换（FFT）相关的操作实现了频域的变换，并最终得到了变换后的频域数据。这个函数将在模块的正向传播过程中被调用，用来对每个通道的数据进行 DCT 变换。

接着，定义了一个名为 `dct_channel_block` 的 PyTorch 模块，它继承自 `nn.Module`。在初始化函数 `__init__` 中，该模块包含了一个神经网络模型，其中通过两个线性层和激活函数构成了一个全连接神经网络。此外，还定义了一个层归一化操作 `dct_norm`，用于对 DCT 变换后的频域数据进行归一化处理。

在前向传播函数 `forward` 中，输入数据 `x` 的形状为 `(B, C, L)`，其中 `B` 表示批次大小，`C` 表示通道数，`L` 表示时间序列的长度。首先对每个通道的数据分别调用之前定义的 `dct` 函数，得到频域数据，并将这些频域数据存储在一个列表中。接着，将列表中的频域数据堆叠起来，得到一个新的张量 `stack_dct`，其形状为 `(B, C, L)`。然后对 `stack_dct` 进行归一化处理，并通过前面定义的全连接神经网络模块 `fc` 对频域数据进行权重调整。最后，将输入数据 `x` 与调整后的权重 `lr_weight` 相乘，得到最终的输出结果。

总之，这个 `dct_channel_block` 模块实现了对输入数据进行离散余弦变换和通道注意力机制的功能，可以被用于深度学习模型中对时间序列数据的处理和特征提取。
"""

def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    # Vc = torch.fft.rfft(v, 1, onesided=False)
    Vc = rfft(v, 1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

class dct_channel_block(nn.Module):
    def __init__(self, channel):
        super(dct_channel_block, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 2, channel, bias=False),
            nn.Sigmoid()
        )

        self.dct_norm = nn.LayerNorm([96], eps=1e-6)  # for lstm on length-wise

    def forward(self, x):
        b, c, l = x.size()  # (B,C,L)
        list = []
        for i in range(c):
            freq = dct(x[:, i, :])
            # print("freq-shape:",freq.shape)
            list.append(freq)

        stack_dct = torch.stack(list, dim=1)
        stack_dct = torch.tensor(stack_dct)
        lr_weight = self.dct_norm(stack_dct)
        lr_weight = self.fc(stack_dct)
        lr_weight = self.dct_norm(lr_weight)

        return x * lr_weight  # result
class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x
class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.adaptive_filter = adaptive_filter

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        threshold = torch.quantile(normalized_energy, self.threshold_param)
        dominant_frequencies = normalized_energy > threshold

        # Initialize adaptive mask
        adaptive_mask = torch.zeros_like(x_fft, device=x_fft.device)
        adaptive_mask[dominant_frequencies] = 1

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x
# LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.icb=ICB(self.input_size, self.hidden_size)
        self.abs=Adaptive_Spectral_Block(self.input_size)
        self.dct = dct_channel_block(channel=self.input_size)
    def forward(self, input_seq):
        input_seq = self.icb(input_seq)
        input_seq=self.abs(input_seq)
        input_seq = dct(input_seq)
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))
        return output, h


class LSTMMain(nn.Module):
    def __init__(self, input_size, output_len, lstm_hidden, lstm_layers, batch_size, device="cpu"):
        super(LSTMMain, self).__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.lstmunit = LSTM(input_size, lstm_hidden, lstm_layers, batch_size, device)
        self.linear = nn.Linear(lstm_hidden, output_len)

    def forward(self, input_seq):
        ula, h_out = self.lstmunit(input_seq)
        out = ula.contiguous().view(ula.shape[0] * ula.shape[1], self.lstm_hidden)
        out = self.linear(out)
        out = out.view(ula.shape[0], ula.shape[1], -1)
        out = out[:, -1, :]
        return out

#TSLANet: Rethinking Transformers for Time Series Representation Learning(ICML 2024)
#https://arxiv.org/pdf/2404.08472
"""
ICB（倒置卷积块）
目的：通过使用一维卷积来处理数据，它包含三个主要步骤。起始和结束使用点集卷积扩展和收缩通道维数，在中间使用核大小为3的卷积捕捉局部模式。
操作：
Conv1：将输入特征从 in_features 扩展到 hidden_features。
Act：应用 GELU 非线性激活函数。
Drop：可选择应用dropout进行正则化。
Conv2：使用三点卷积核处理扩展的特征，捕捉局部模式。
Conv3：将特征从 hidden_features 缩减回 in_features。
元素级操作：在最终缩减操作前，结合conv1和conv2的输出进行乘法和加法运算。

自适应频谱块（Adaptive_Spectral_Block）
目的：该模块旨在处理频域中的数据，根据频率分量的能量来自适应地调制或强化特定的频率分量。
操作：
FFT：首先将输入时间序列转换到频域。
自适应掩膜：计算每个频率分量的能量，并根据百分位阈值创建一个掩膜以识别主导频率。
复数权重：在应用自适应掩膜之前和之后，变为复数值的权重被应用在频域表示上，用以调节信号的频谱属性。
IFFT：将数据转换回时域。
TSLANet层
综合：这一层结合了自适应频谱块（ASB）和倒置卷积块（ICB），通过外部布尔标志（args.ICB 和 args.ASB）决定是否对每个块应用数据。
顺序处理：
Norm1：首先规范输入数据。
ASB/ICB 应用：根据标志，它要么：
通过ASB然后ICB来传递数据。
只通过其中一个块来处理数据。
如果不应用任何块，直接通过数据而不做改变。
Norm2 和 ICB：对于ICB处理，数据在通过ICB前再次规范化。
"""
