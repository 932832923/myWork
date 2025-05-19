import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from module import KAN
from sklearn.metrics import r2_score
from module import *



class KAN_TCN_transformer(nn.Module):

    def __init__(self, indim, dim, outdim,KAN_dim,num_inputs,nb_unites,num_channels,kernel_size,dropout):
        super(KAN_TCN_transformer, self).__init__()

        self.KAN0 = KAN([indim, KAN_dim, dim])
        self.KAN1 = KAN([dim, KAN_dim, outdim])

        self.conv1 = nn.Conv1d(1,num_inputs,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(num_inputs,1,kernel_size=3,padding=1)

        self.transformer = TCN_transfomer(nb_unites=nb_unites,num_inputs=num_inputs,num_channels=num_channels,kernel_size=kernel_size, dropout=dropout)

    def forward(self, x):
        

        
        x = self.KAN0(x)

        print(x.shape)
        
        #x = x.unsqueeze(1)

        x = self.conv1(x)
        
        print(x.shape)

        x = self.transformer(x)

        #x = x.squeeze(1)

        x = self.KAN1(x)

        return x

class KAN_net(nn.Module):

    def __init__(self, indim, dim, outdim,KAN_dim,num_inputs):
        super(KAN_net, self).__init__()

        self.KAN0 = KAN([indim, KAN_dim, dim])
        self.KAN1 = KAN([dim, KAN_dim, outdim])

    def forward(self, x):
        
        x = self.KAN0(x)

        x = self.KAN1(x)

        return x

class lineargresion(nn.Module):

    def __init__(self, indim, dim, outdim):
        super(lineargresion, self).__init__()

        self.linear1 = nn.Linear(indim,  dim)
        self.linear2 = nn.Linear( dim,outdim)

    def forward(self, x):
        
        x = self.linear1(x)

        x = self.linear2(x)

        return x
    
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=1, 
                            batch_first=True)  # 输入数据的批次维度是第一维
        # 将 LSTM 输出转换为所需输出维度
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x 的形状应为 [32, 8]
        # LSTM 期望的输入形状为 [batch_size, seq_len, input_size]
        # 由于没有时间序列，我们将 seq_len 设置为 1
        x = x.unsqueeze(1)  # 现在 x 的形状是 [32, 1, 8]
        
        # 前向传播 LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out 的形状是 [32, 1, hidden_dim]
        
        # 取出 LSTM 最后时间步的输出用于预测
        last_time_step = lstm_out[:, -1, :]  # 形状是 [32, hidden_dim]
        
        # 通过全连接层进行预测
        predictions = self.fc(last_time_step)  # 预测结果的形状是 [32, 2]
        return predictions