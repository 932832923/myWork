import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Transformer_EncDec import Encoder, EncoderLayer
from model.SelfAttention_Family import FullAttention, AttentionLayer
from model.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.fc = nn.Linear(7*1, 1)
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x, mask=None):
        x_enc, x_mark_enc, x_dec, x_mark_dec = x,x,x,x
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        out = self.fc(dec_out[:, -self.pred_len:, :].reshape(-1, 1*7))
        return out


# class Configs:
#     def __init__(self):
#         self.seq_len = 4           # 输入序列长度
#         self.pred_len = 1          # 预测长度
#         self.output_attention = True  # 是否输出注意力
#         self.use_norm = True          # 是否使用归一化
#         self.d_model = 7             # 模型的特征维度
#         self.embed = 'timeF'         # 嵌入方式
#         self.freq = 'h'              # 时间频率
#         self.dropout = 0.1            # dropout概率
#         self.class_strategy = None    # 类别策略（可根据需要定义）
#         self.factor = 5               # 注意力因子
#         self.n_heads = 1              # 注意力头数
#         self.e_layers = 3             # 编码器层数
#         self.d_ff = 128               # 前馈网络的维度
#         self.activation = 'gelu'      # 激活函数

# # 创建配置
# configs = Configs()

# # 创建一个 iTransformer 模型
# model = Model(configs)

# x_enc = torch.rand(32, configs.seq_len, configs.d_model)  # 输入编码

# # 前向传播
# output = model(x_enc)

# # 打印输出的形状
# print("Output shape:", output.shape)  
