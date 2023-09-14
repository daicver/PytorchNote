
# https://pytorch.org/docs/stable/nn.html
# 常见网络层的参数和含义

import torch

# 卷积
torch.nn.Conv1d(in_channels,              # 输入通道数
                out_channels,             # 输出通道数
                kernel_size,              # 卷积核大小
                stride=1,                 # 步长
                padding=0,                # 边界填充大小
                dilation=1,               # 扩张大小
                groups=1,                 # 输入通道和输出通道的连接分组
                bias=True,                # 是否需要偏置
                padding_mode='zeros',     # 边界填充模式
                device=None,              # 设备
                dtype=None)               # 数据类型
torch.nn.Conv2d('same as Conv1d')
torch.nn.Conv3d('same as Conv1d')
# 输入特征尺寸:   B*H1*W1*C1
# 卷积核尺寸:     k*k
# 输出特征尺寸:   B*H2*W2*C2
# 输出特征尺寸计算：
# W2 = (W1+2P-k)/S + 1 # 一般向下取整
# H2 = (H1+2P-k)/S + 1 # 一般向下取整
# 卷积层的参数量： (k*k*C1)*C2 + C2
# 卷积层的计算量： k*k*(H1*W1)*(C2)

# 转置卷积
torch.nn.ConvTranspose1d(in_channels, 
                         out_channels, 
                         kernel_size, 
                         stride=1, 
                         padding=0, 
                         output_padding=0, 
                         groups=1, 
                         bias=True, 
                         dilation=1, 
                         padding_mode='zeros', 
                         device=None, 
                         dtype=None)
torch.nn.ConvTranspose2d('same as ConvTranspose1d')
torch.nn.ConvTranspose3d('same as ConvTranspose1d')

# 池化
torch.nn.MaxPool1d(kernel_size, 
                   stride=None, 
                   padding=0, 
                   dilation=1, 
                   return_indices=False, 
                   ceil_mode=False)
torch.nn.MaxPool2d('same as MaxPool1d')
torch.nn.MaxPool3d('same as MaxPool1d')
torch.nn.AvgPool1d('same as MaxPool1d')
torch.nn.AvgPool2d('same as MaxPool1d')
torch.nn.AvgPool3d('same as MaxPool1d')

# 归一化
torch.nn.BatchNorm1d(num_features,                  # 特征的通道数
                     eps=1e-05,                     # 为了稳定加在分母上的极小数
                     momentum=0.1,                  # 用于 running_mean 和 running_var 计算的值
                     affine=True,                   # 是否具有可学习的纺射参数
                     track_running_stats=True,      # 是否跟踪学习 running_mean 和 running_var
                     device=None, 
                     dtype=None)
torch.nn.BatchNorm2d()
torch.nn.BatchNorm3d()
torch.nn.GroupNorm()
torch.nn.SyncBatchNorm()
torch.nn.InstanceNorm1d()
torch.nn.InstanceNorm2d()
torch.nn.InstanceNorm3d()
torch.nn.LayerNorm()

# 线性层/全连接层
torch.nn.Linear(in_features,        # 输入通道数
                out_features,       # 输出通道数
                bias=True,          # 权重参数是否有偏置
                device=None, 
                dtype=None)

# 线性激活层
torch.nn.ReLU()
torch.nn.PReLU()
torch.nn.ReLU6()
torch.nn.RReLU()
torch.nn.GELU()
torch.nn.Sigmoid()

# 非线形激活层
torch.nn.Softmax()

# 丢失连接层
torch.nn.Dropout(p=0.5,                 # 元素归零的概率            
                 inplace=False)         # 是否就地执行操作

# mean absolute error (MAE)
torch.nn.L1Loss(size_average=None, 
                reduce=None, 
                reduction='mean')

# mean squared error (squared L2 norm) 
torch.nn.MSELoss(size_average=None, 
                 reduce=None, 
                 reduction='mean')

# cross entropy loss
torch.nn.CrossEntropyLoss(weight=None, 
                          size_average=None, 
                          ignore_index=- 100, 
                          reduce=None, 
                          reduction='mean', 
                          label_smoothing=0.0)
