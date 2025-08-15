from models.convlstm_torch import ConvLSTM
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
import torch.nn.functional as F



class SWE_NET(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, kernel_size=(3, 3), height=390, width=348, dropout_rate=0.3):
        super(SWE_NET, self).__init__()

        self.convlstm1 = ConvLSTM(input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  kernel_size=kernel_size,
                                  num_layers=1,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=True)

        self.bn1 = nn.BatchNorm3d(hidden_dim)
        # self.drop1 = nn.Dropout3d(dropout_rate)

        self.convlstm2 = ConvLSTM(input_dim=hidden_dim,
                                  hidden_dim=hidden_dim,
                                  kernel_size=kernel_size,
                                  num_layers=1,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=True)

        self.bn2 = nn.BatchNorm3d(hidden_dim)
        # self.drop2 = nn.Dropout3d(dropout_rate)

        self.convlstm3 = ConvLSTM(input_dim=hidden_dim,
                                  hidden_dim=hidden_dim,
                                  kernel_size=kernel_size,
                                  num_layers=1,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=False)

        self.bn3 = nn.BatchNorm2d(hidden_dim)
        # self.drop3 = nn.Dropout2d(dropout_rate)

        self.conv_final = nn.Conv2d(in_channels=hidden_dim, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: (B, T, C, H, W)
        out1, _ = self.convlstm1(x)
        out1 = out1[0]  # (B, T, C, H, W)
        out1 = out1.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        out1 = self.bn1(out1)
        out1 = out1.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
    
        out2, _ = self.convlstm2(out1)
        out2 = out2[0]
        out2 = out2.permute(0, 2, 1, 3, 4)
        #out2 = self.drop2(self.bn2(out2))
        out2 = self.bn2(out2)
        out2 = out2.permute(0, 2, 1, 3, 4)
    
        out3, _ = self.convlstm3(out2)
        out3 = out3[0][:, -1]  # (B, C, H, W)
        out3 = self.bn3(out3) # (B, C, H, W)
    
        out_final = self.conv_final(out3)
        return self.sigmoid(out_final)
