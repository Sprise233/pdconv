from typing import Union, List, Tuple, Type

import torch
import torch.nn as nn

from models.PDCN.blocks.attentions import efficient_attention_fusion
from models.lg_net.blocks.attentions import ChannelAttention, PositionAttention
from models.utils.helper import avg_pool
from models.utils.simple_conv_blocks import StackedConvBlocks


class Fusion1(nn.Module):
    # Exchanging Dual-Encoder–Decoder: A New Strategy for Change Detection With Semantic Guidance and Spatial Localization
    def __init__(self,
                 input_channels: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 add_channel_att: bool = False,
                 add_position_att: bool = False
                 ):
        super().__init__()
        self.add_channel_att = add_channel_att
        self.add_position_att = add_position_att

        if add_channel_att:
            self.conv1 = StackedConvBlocks(1, nn.Conv3d, input_channels * 2, input_channels, (1, 1, 1),
                                           1, conv_bias, None,
                                           None, None, None, None, None)

            self.conv2 = StackedConvBlocks(1, nn.Conv3d, input_channels * 2, input_channels, (1, 1, 1),
                                           1, conv_bias, None,
                                           None, None, None, None, None)

        if add_position_att:
            self.conv3 = StackedConvBlocks(1, nn.Conv3d, 2, 1, (7, 7, 7),
                                           1, conv_bias, None,
                                           None, None, None, None, None)

            self.conv4 = StackedConvBlocks(1, nn.Conv3d, 2, 1, (7, 7, 7),
                                           1, conv_bias, None,
                                           None, None, None, None, None)

    def forward(self, x1, x2):
        position_att1, position_att2, channel_att1, channel_att2 = None, None, None, None
        if self.add_position_att:
            position_avg_x1 = avg_pool(x1, keepdim=True, dim=(1,))
            position_avg_x2 = avg_pool(x2, keepdim=True, dim=(1,))

            position_feature = torch.cat((position_avg_x1, position_avg_x2), dim=1)
            position_feature1 = self.conv3(position_feature)
            position_feature2 = self.conv4(position_feature)
            position_att1, position_att2 = efficient_attention_fusion(position_feature1, position_feature2)

        if self.add_channel_att:
            channel_avg_x1 = avg_pool(x1, keepdim=True, dim=(2, 3, 4))
            channel_avg_x2 = avg_pool(x2, keepdim=True, dim=(2, 3, 4))

            channel_feature = torch.cat((channel_avg_x1, channel_avg_x2), dim=1)
            channel_feature1 = self.conv1(channel_feature)
            channel_feature2 = self.conv2(channel_feature)
            channel_att1, channel_att2 = efficient_attention_fusion(channel_feature1, channel_feature2)

        if self.add_channel_att == self.add_position_att == False:
            return x1 + x2
        elif self.add_channel_att == True and self.add_position_att == False:
            return x1 * channel_att1 + x2 * channel_att2 + x1 + x2
        elif self.add_channel_att == False and self.add_position_att == True:
            return x1 * position_att1 + x2 * position_att2 + x1 + x2
        elif self.add_channel_att == True and self.add_position_att == True:
            return x1 * (position_att1 + channel_att1) + x2 * (position_att2 + channel_att2) + x1 + x2
        else:
            return None

    def compute_conv_feature_map_size(self, input_size):
        output = 0
        output += self.conv_block11.compute_conv_feature_map_size(input_size)
        output += self.conv_block12.compute_conv_feature_map_size(input_size)
        return output


class AG(nn.Module):
    def __init__(self,
                 input_channels: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 add_channel_att: bool = False,
                 add_position_att: bool = False
                 ):
        super().__init__()
        self.conv1 = StackedConvBlocks(1, nn.Conv3d, input_channels, input_channels, (1, 1, 1),
                                       1, conv_bias, norm_op,
                                       norm_op_kwargs, None, None, nonlin, nonlin_kwargs)

        self.conv2 = StackedConvBlocks(1, nn.Conv3d, input_channels, input_channels, (1, 1, 1),
                                       1, conv_bias, norm_op,
                                       norm_op_kwargs, None, None, nonlin, nonlin_kwargs)

        self.conv3 = StackedConvBlocks(1, nn.Conv3d, input_channels, 1, (1, 1, 1),
                                       1, conv_bias, None,
                                       None, None, None, None, None)

        self.act = nonlin(**nonlin_kwargs)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        fusion_feature = self.conv3(self.act(x1 + x2))
        att = torch.sigmoid(fusion_feature)
        return x1 * att + x1

    def compute_conv_feature_map_size(self, input_size):
        output = 0
        output += self.conv_block11.compute_conv_feature_map_size(input_size)
        output += self.conv_block12.compute_conv_feature_map_size(input_size)
        return output


class EFF(nn.Module):
    # SUnet: A multi-organ segmentation network based on multiple attention
    def __init__(self,
                 input_channels: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 add_channel_att: bool = False,
                 add_position_att: bool = False
                 ):
        super().__init__()
        norm_op = nn.BatchNorm3d
        norm_op_kwargs = {}
        nonlin = nn.ReLU
        nonlin_kwargs = {}

        self.ag = AG(input_channels, norm_op=norm_op, conv_bias=conv_bias, norm_op_kwargs=norm_op_kwargs,
                     nonlin_kwargs=nonlin_kwargs, nonlin=nonlin)
        self.channel_attention = ChannelAttention(channels=input_channels, conv_op=nn.Conv3d, rd_ratio=1. / 4,
                                                  nonlin_kwargs=nonlin_kwargs, nonlin=nonlin)
        self.position_attention = PositionAttention()
        self.act = nonlin(**nonlin_kwargs)

    def forward(self, x1, x2):
        return self.position_attention(self.channel_attention(self.ag(x1, x2)))

    def compute_conv_feature_map_size(self, input_size):
        output = 0
        output += self.conv_block11.compute_conv_feature_map_size(input_size)
        output += self.conv_block12.compute_conv_feature_map_size(input_size)
        return output


class FusionAdd(nn.Module):
    # Exchanging Dual-Encoder–Decoder: A New Strategy for Change Detection With Semantic Guidance and Spatial Localization
    def __init__(self,
                 input_channels: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 add_channel_att: bool = False,
                 add_position_att: bool = False
                 ):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2

    def compute_conv_feature_map_size(self, input_size):
        output = 0
        return output


class MS_CAM(nn.Module):
    def __init__(self,
                 input_channels: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None
                 ):
        super().__init__()
        self.conv11 = StackedConvBlocks(1, nn.Conv3d, input_channels, input_channels // 4, (1, 1, 1),
                                        1, conv_bias, None,
                                        None, None, None, nonlin, nonlin_kwargs)
        self.conv12 = StackedConvBlocks(1, nn.Conv3d, input_channels // 4, input_channels, (1, 1, 1),
                                        1, conv_bias, None,
                                        None, None, None, None, None)
        self.conv21 = StackedConvBlocks(1, nn.Conv3d, input_channels, input_channels // 4, (1, 1, 1),
                                        1, conv_bias, None,
                                        None, None, None, nonlin, nonlin_kwargs)
        self.conv22 = StackedConvBlocks(1, nn.Conv3d, input_channels // 4, input_channels, (1, 1, 1),
                                        1, conv_bias, None,
                                        None, None, None, None, None)

    def forward(self, x):
        x_avgpool = avg_pool(x, dim=(2, 3, 4), keepdim=True)
        x_avgpool = self.conv12(self.conv11(x_avgpool))
        x = self.conv22(self.conv21(x))
        x = x + x_avgpool
        return torch.sigmoid(x)


class iFFN(nn.Module):
    def __init__(self,
                 input_channels: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 add_channel_att: bool = False,
                 add_position_att: bool = False
                 ):
        super().__init__()
        self.MS_CAM1 = MS_CAM(input_channels, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs)
        self.MS_CAM2 = MS_CAM(input_channels, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs)

    def forward(self, x1, x2):
        att1 = self.MS_CAM1(x1 + x2)
        x11 = x1 * att1
        x12 = x2 * (1 - att1)
        att2 = self.MS_CAM2(x11 + x12)
        return x1 * att2 + x2 * (1 - att2)

    def compute_conv_feature_map_size(self, input_size):
        output = 0
        return output
