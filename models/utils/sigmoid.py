import torch.nn as nn
import torch

class GLEU(nn.Module):
    def __init__(self, num_channels, dim='2d'):
        super(GLEU, self).__init__()
        self.dim = dim
        # 可学习的仿射参数 gamma 和 beta
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        if dim == '3d':
            self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))

    def forward(self, x):
        # 根据维度选择合适的范数计算
        if self.dim == '3d':
            gx = torch.norm(x, p=2, dim=(-3, -2, -1), keepdim=True)  # 对 H, W, D 计算范数
        elif self.dim == '2d':
            gx = torch.norm(x, p=2, dim=(-2, -1), keepdim=True)  # 对 H, W 计算范数

        # 归一化 gx
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)

        # 应用仿射变换，并加上残差连接
        return self.gamma * (x * nx) + self.beta + x