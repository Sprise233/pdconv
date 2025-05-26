from typing import Union, Type

import torch
import torch.nn as nn

from models.utils.PDConv import PDConv
from models.utils.helper import max_pool, avg_pool
from utils.vis_attention import vis_feature_map


class SliceAttention(nn.Module):
    def __init__(self,
                 num_slice,
                 channels,
                 has_mlp=True,
                 has_norm=True,
                 rate=4,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,

                 ):
        super().__init__()
        if has_mlp:
            if nonlin is not None:
                self.mlp = nn.Sequential(
                    nn.Linear(num_slice, int(num_slice // rate)),
                    nonlin(**nonlin_kwargs),
                    nn.Linear(int(num_slice // rate), num_slice)
                )
            else:
                self.mlp = nn.Sequential(
                    nn.Linear(num_slice, num_slice)
                )

        self.has_norm = has_norm
        self.has_mlp = has_mlp
        if norm_op is not None and has_norm:
            self.norm1 = norm_op(channels, **norm_op_kwargs)
            self.norm2 = norm_op(channels, **norm_op_kwargs)
        else:
            self.norm1, self.norm2 = None, None

    def forward(self, feature_map1, feature_map2):
        assert feature_map1.shape == feature_map2.shape, f'特征图大小不同，大小为：{feature_map1.shape}和{feature_map2.shape}'

        if None not in [self.norm1, self.norm2] and self.has_norm:
            feature_map1 = self.norm1(feature_map1)
            feature_map2 = self.norm2(feature_map2)

        # 获取张量的维度
        b, c, l, h, w = feature_map1.shape
        # 计算特征差异图
        feature_different_map = (feature_map1 - feature_map2) / torch.sqrt(torch.tensor(int(c * h * w)))

        if self.has_mlp:
            # 计算 q
            q = torch.norm(feature_different_map, p=2, dim=(1, 3, 4), keepdim=False)
            att = self.mlp(q)
            # 调整 att 维度
            att = att.unsqueeze(1).unsqueeze(3).unsqueeze(4)
        else:
            att = torch.norm(feature_different_map, p=2, dim=(1, 3, 4), keepdim=True)

        # 通过 Sigmoid 函数
        att = torch.sigmoid(att)

        return att

def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    """
    This function is taken from the timm package (https://github.com/rwightman/pytorch-image-models/blob/b7cb8d0337b3e7b50516849805ddb9be5fc11644/timm/models/layers/helpers.py#L25)
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class ChannelAttention(nn.Module):
    """
    This class is taken from the timm package (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/squeeze_excite.py)
    and slightly modified so that the convolution type can be adapted.

    SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, channels,
            conv_op,
            rd_ratio=1. / 16,
            rd_channels=None,
            rd_divisor=8,
            add_maxpool=False,
            nonlin: Union[None, Type[torch.nn.Module]] = nn.ReLU,
            nonlin_kwargs: dict = {},
            gate_layer=nn.Sigmoid):
        super(ChannelAttention, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = conv_op(channels, rd_channels, kernel_size=1, bias=True)
        self.act = nonlin(**nonlin_kwargs)
        self.fc2 = conv_op(rd_channels, channels, kernel_size=1, bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        x_se = 0

        mean_se = x.mean((2, 3, 4), keepdim=True)

        mean_se = self.fc1(mean_se)
        mean_se = self.act(mean_se)
        mean_se = self.fc2(mean_se)

        x_se = mean_se
        att = self.gate(x_se)

        if self.add_maxpool:
            # experimental codepath, may remove or change
            max_se = x.amax((2, 3, 4), keepdim=True)
            max_se = self.fc1(max_se)
            max_se = self.act(max_se)
            max_se = self.fc2(max_se)

            att = self.gate(max_se) + att

        return x * att



class ChannelAttentionUpgrade(nn.Module):
    def __init__(self,
                 channels,
                 bias,
                 nonlin: Union[None, Type[torch.nn.Module]],
                 nonlin_kwargs: dict,
                 ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=bias),
            nonlin(**nonlin_kwargs),
            nn.ConvTranspose2d(in_channels=1, out_channels=2, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0),
                               bias=bias)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        avg_x1 = avg_pool(x1, keepdim=True, dim=(3, 4))
        avg_x2 = avg_pool(x2, keepdim=True, dim=(3, 4))

        avg_x1 = avg_x1.squeeze(-1).squeeze(-1)
        avg_x1 = avg_x1.unsqueeze(1)

        avg_x2 = avg_x2.squeeze(-1).squeeze(-1)
        avg_x2 = avg_x2.unsqueeze(1)

        avg_x = torch.cat((avg_x1, avg_x2), dim=1)

        att_avg = self.mlp(avg_x)

        att = att_avg

        att = att.unsqueeze(-1).unsqueeze(-1).squeeze(1)
        att1 = self.sigmoid(att[:, 0, :, :, :])
        att2 = self.sigmoid(att[:, 1, :, :, :])
        return att1, att2

def efficient_attention_fusion(x1, x2):
    b, c, l, h, w = x1.shape
    feature1 = x1.flatten(2).unsqueeze(-2)  # [b, c, l*h*w]
    feature2 = x2.flatten(2).unsqueeze(-2)  # [b, c, l*h*w]

    feature = torch.cat((feature1, feature2), dim=-2)  # [b, c, 2, l*h*w]
    att = torch.softmax(feature, dim=-2)  # [b, c, 2, l*h*w]

    att1 = att[:, :, 0, :]  # [b, c, l*h*w] each
    att2 = att[:, :, 1, :]  # [b, c, l*h*w] each

    att1 = att1.view(b, c, l, h, w)  # [b, c, l, h, w]
    att2 = att2.view(b, c, l, h, w)  # [b, c, l, h, w]

    # fusion_feature = x1 * att1 + x2 * att2  # [b, c, l, h, w]
    # fusion_feature = torch.add(x1, att1.mul(x1))
    return att1, att2

def calculate_balance_weight(tensor):
    """
    计算每个体素点的均衡度并给出权重，均衡度较高则权重较大。

    参数：
        tensor: 一个形状为(b, c, l, h, w)的张量，其中b为batch size，c为通道数，l、h、w为空间维度。

    返回：
        weight: 一个形状为(b, l, h, w)的张量，每个位置为对应体素点的权重。
    """
    # 计算每个体素点在所有通道上的方差
    # 输入 tensor 形状为 (b, c, l, h, w)
    # 我们需要在通道维度上计算方差
    tensor_var = torch.var(tensor, dim=1, unbiased=False, keepdim=True)  # 计算每个体素点通道上的方差，形状为 (b, l, h, w)

    # 方差较小意味着更均衡，所以我们用方差的倒数作为权重
    weight = 1 / (tensor_var + 1e-6)  # 防止除以零，加上一个小的常数 1e-6

    # 归一化权重（可以根据需求选择是否归一化）
    weight = (weight - weight.min()) / (weight.max() - weight.min())

    return weight

class PositionAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.conv = PDConv(input_channels=2, output_channels=1, kernel_size=7, initial_stride=1)


    def forward(self, x):
        max_x = max_pool(x, keepdim=True, dim=(1,))
        avg_x = avg_pool(x, keepdim=True, dim=(1,))
        return self.sigmoid(self.conv(torch.concat((max_x, avg_x), dim=1))) * x


class SliceAttentionLocal(nn.Module):
    def __init__(self,
                 num_slice,
                 channels,
                 has_mlp=True,
                 has_norm=True,
                 rate=4,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 is_single_attention_weight: bool = True
                 ):
        super().__init__()
        if has_mlp:
            if nonlin is not None:
                # 判定是否是双层mlp，否则只是线性层
                self.mlp = nn.Sequential(
                    nn.Linear(num_slice, int(num_slice // rate)),
                    nonlin(**nonlin_kwargs),
                )
                if is_single_attention_weight:
                    self.mlp_output = nn.Sequential(
                        nn.Linear(int(num_slice // rate), num_slice)
                    )
                else:
                    self.mlp_output1 = nn.Sequential(
                        nn.Linear(int(num_slice // rate), num_slice)
                    )
                    self.mlp_output2 = nn.Sequential(
                        nn.Linear(int(num_slice // rate), num_slice)
                    )

            else:
                self.mlp = nn.Sequential(
                    nn.Linear(num_slice, num_slice)
                )

        self.has_norm = has_norm
        self.has_mlp = has_mlp
        self.is_single_attention_weight = is_single_attention_weight

        if norm_op is not None and has_norm:
            self.norm1 = norm_op(channels, **norm_op_kwargs)
            self.norm2 = norm_op(channels, **norm_op_kwargs)
        else:
            self.norm1, self.norm2 = None, None

        self.nonlin = nonlin

    def forward(self, feature_map1, feature_map2):
        assert feature_map1.shape == feature_map2.shape, f'特征图大小不同，大小为：{feature_map1.shape}和{feature_map2.shape}'

        if None not in [self.norm1, self.norm2] and self.has_norm:
            feature_map1 = self.norm1(feature_map1)
            feature_map2 = self.norm2(feature_map2)

        # 获取张量的维度
        b, c, l, h, w = feature_map1.shape
        # 计算特征差异图
        feature_different_map = (feature_map1 - feature_map2) / torch.sqrt(torch.tensor(int(c * h * w)))

        if self.has_mlp:
            # 计算 q
            q = torch.norm(feature_different_map, p=2, dim=(1, 3, 4), keepdim=False)
            # vis_feature_map(feature_different_map[0, :, -1, :, :], n_components=8, save_path=f'./feature_map_vis_feature_different_map.png')
            # print(q)
            att = self.mlp(q)
            if self.nonlin is not None:
                if self.is_single_attention_weight:
                    att = self.mlp_output(att)
                    # print(att)
                    # 调整 att 维度
                    att = att.unsqueeze(1).unsqueeze(3).unsqueeze(4)
                    # 通过 Sigmoid 函数
                    att = torch.sigmoid(att)
                    return att
                else:
                    att1 = self.mlp_output1(att)
                    att2 = self.mlp_output2(att)

                    # 通过 Sigmoid 函数
                    att1 = att1.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                    att2 = att2.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                    att1 = torch.sigmoid(att1)
                    att2 = torch.sigmoid(att2)
                    return att1, att2
            else:
                att = att.unsqueeze(1).unsqueeze(3).unsqueeze(4)
                att = torch.sigmoid(att)
                return att

        else:
            att = torch.norm(feature_different_map, p=2, dim=(1, 3, 4), keepdim=True)
            # 通过 Sigmoid 函数
            att = torch.sigmoid(att)
            return att


class ChannelAttentionLocal(nn.Module):
    def __init__(self,
                 channels,
                 rate=4,
                 has_mlp=True,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 is_single_attention_weight: bool = True
                 ):
        super().__init__()
        if has_mlp:
            if nonlin is not None:
                self.mlp = nn.Sequential(
                    nn.Conv3d(channels, int(channels // rate), 1, bias=False),
                    nonlin(**nonlin_kwargs)
                )
                if is_single_attention_weight:
                    self.mlp_output = nn.Sequential(
                        nn.Conv3d(int(channels // rate), channels, 1, bias=False)
                    )
                else:
                    self.mlp_output1 = nn.Sequential(
                        nn.Conv3d(int(channels // rate), channels, 1, bias=False)
                    )
                    self.mlp_output2 = nn.Sequential(
                        nn.Conv3d(int(channels // rate), channels, 1, bias=False)
                    )
            else:
                self.mlp = nn.Sequential(
                    nn.Conv3d(channels, channels, 1, bias=True)
                )
        self.has_mlp = has_mlp
        self.is_single_attention_weight = is_single_attention_weight

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x_l = avg_pool(x, keepdim=True, dim=(3, 4))
        avg_x_h = avg_pool(x, keepdim=True, dim=(2, 4))
        avg_x_w = avg_pool(x, keepdim=True, dim=(2, 3))

        avg_x_l = torch.permute(avg_x_l, (0, 1, 3, 4, 2))
        avg_x_h = torch.permute(avg_x_h, (0, 1, 2, 4, 3))
        avg_x_w = torch.permute(avg_x_w, (0, 1, 2, 3, 4))

        avg_x = torch.concat((avg_x_w, avg_x_h, avg_x_l), dim=-1)

        if self.has_mlp:
            att = self.mlp(avg_x)
            if self.is_single_attention_weight:
                att = self.mlp_output(att)
                att = self.sigmoid(att)
                return att
            else:
                att1 = self.mlp_output1(att)
                att2 = self.mlp_output2(att)
                att1 = avg_pool(att1, dim=(-1,), keepdim=True)
                att2 = avg_pool(att2, dim=(-1,), keepdim=True)
                att1 = self.sigmoid(att1)
                att2 = self.sigmoid(att2)
                return att1, att2
        else:
            return self.sigmoid(avg_x)


class PositionAttentionLocal(nn.Module):
    def __init__(self,
                 is_single_attention_weight: bool = True):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=(1, 7, 7), stride=1, padding=(0, 3, 3))

    def forward(self, x):
        max_x = max_pool(x, keepdim=True, dim=(1, 2))
        avg_x = avg_pool(x, keepdim=True, dim=(1, 2))
        return self.sigmoid(self.conv(torch.concat((max_x, avg_x), dim=1)))
