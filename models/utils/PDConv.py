from typing import Union, List, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.dropout import _DropoutNd

from models.utils.BasicConv import BasicConv, BasicConv3D
from models.utils.helper import maybe_convert_scalar_to_list
from models.utils.simple_conv_blocks import StackedConvBlocks





class PDConv_A(BasicConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        l, h, w = self.kernel_size

        self.conv_block1 = BasicConv3D(self.input_channels,
                                       self.output_channels,
                                       self.initial_stride,
                                       self.conv_bias,
                                       None,
                                       None,
                                       self.dropout_op,
                                       self.dropout_op_kwargs,
                                       None,
                                       None,
                                       is_dw_conv=self.is_dw_conv,
                                       kernel_size=(1, h, w))

        self.conv_block2 = BasicConv3D(self.input_channels,
                                       self.output_channels,
                                       self.initial_stride,
                                       self.conv_bias,
                                       None,
                                       None,
                                       self.dropout_op,
                                       self.dropout_op_kwargs,
                                       None,
                                       None,
                                       is_dw_conv=self.is_dw_conv,
                                       kernel_size=(l, 1, 1))


    def forward(self, x):

        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(x)

        final_feature = conv1 + conv2

        if self.has_final_process:
            return self.final_process(final_feature)
        else:
            return final_feature


    def compute_conv_feature_map_size(self, input_size):
        output = 0
        output += self.conv_block1.compute_conv_feature_map_size(input_size)
        output += self.conv_block2.compute_conv_feature_map_size(input_size)
        return output

class PDConv_D(BasicConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        l, h, w = self.kernel_size

        self.conv_block1 = BasicConv3D(self.input_channels,
                                       self.output_channels,
                                       self.initial_stride,
                                       self.conv_bias,
                                       None,
                                       None,
                                       self.dropout_op,
                                       self.dropout_op_kwargs,
                                       None,
                                       None,
                                       is_dw_conv=self.is_dw_conv,
                                       kernel_size=(l, 1, w))

        self.conv_block2 = BasicConv3D(self.input_channels,
                                       self.output_channels,
                                       self.initial_stride,
                                       self.conv_bias,
                                       None,
                                       None,
                                       self.dropout_op,
                                       self.dropout_op_kwargs,
                                       None,
                                       None,
                                       is_dw_conv=self.is_dw_conv,
                                       kernel_size=(1, h, 1))


    def forward(self, x):

        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(x)

        final_feature = conv1 + conv2

        if self.has_final_process:
            return self.final_process(final_feature)
        else:
            return final_feature


    def compute_conv_feature_map_size(self, input_size):
        output = 0
        output += self.conv_block1.compute_conv_feature_map_size(input_size)
        output += self.conv_block2.compute_conv_feature_map_size(input_size)
        return output

class PDConv_B(BasicConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        l, h, w = self.kernel_size

        self.conv_block1 = BasicConv3D(self.input_channels,
                                       self.output_channels,
                                       self.initial_stride,
                                       self.conv_bias,
                                       None,
                                       None,
                                       self.dropout_op,
                                       self.dropout_op_kwargs,
                                       None,
                                       None,
                                       is_dw_conv=self.is_dw_conv,
                                       kernel_size=(1, h, w))

        self.conv_block2 = BasicConv3D(self.input_channels,
                                       self.output_channels,
                                       self.initial_stride,
                                       self.conv_bias,
                                       None,
                                       None,
                                       self.dropout_op,
                                       self.dropout_op_kwargs,
                                       None,
                                       None,
                                       is_dw_conv=self.is_dw_conv,
                                       kernel_size=(h, 1, w))

        self.conv_block3 = BasicConv3D(self.input_channels,
                                       self.output_channels,
                                       self.initial_stride,
                                       self.conv_bias,
                                       None,
                                       None,
                                       self.dropout_op,
                                       self.dropout_op_kwargs,
                                       None,
                                       None,
                                       is_dw_conv=self.is_dw_conv,
                                       kernel_size=(l, h, 1))

    def forward(self, x):

        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(x)
        conv3 = self.conv_block3(x)

        # final_feature = efficient_attention_fusion(conv1, conv2)
        final_feature = conv1 + conv2 + conv3

        if self.has_final_process:
            return self.final_process(final_feature)
        else:
            return final_feature

    def compute_conv_feature_map_size(self, input_size):
        output = 0
        output += self.conv_block1.compute_conv_feature_map_size(input_size)
        output += self.conv_block2.compute_conv_feature_map_size(input_size)
        output += self.conv_block3.compute_conv_feature_map_size(input_size)
        return output

class PDConv_C(BasicConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        l, h, w = self.kernel_size

        self.conv_block1 = BasicConv3D(self.input_channels,
                                       self.output_channels,
                                       self.initial_stride,
                                       self.conv_bias,
                                       None,
                                       None,
                                       self.dropout_op,
                                       self.dropout_op_kwargs,
                                       None,
                                       None,
                                       is_dw_conv=self.is_dw_conv,
                                       kernel_size=(1, 1, w))

        self.conv_block2 = BasicConv3D(self.input_channels,
                                       self.output_channels,
                                       self.initial_stride,
                                       self.conv_bias,
                                       None,
                                       None,
                                       self.dropout_op,
                                       self.dropout_op_kwargs,
                                       None,
                                       None,
                                       is_dw_conv=self.is_dw_conv,
                                       kernel_size=(1, h, 1))

        self.conv_block3 = BasicConv3D(self.input_channels,
                                       self.output_channels,
                                       self.initial_stride,
                                       self.conv_bias,
                                       None,
                                       None,
                                       self.dropout_op,
                                       self.dropout_op_kwargs,
                                       None,
                                       None,
                                       is_dw_conv=self.is_dw_conv,
                                       kernel_size=(l, 1, 1))

    def forward(self, x):

        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(x)
        conv3 = self.conv_block3(x)

        # final_feature = efficient_attention_fusion(conv1, conv2)
        final_feature = conv1 + conv2 + conv3

        if self.has_final_process:
            return self.final_process(final_feature)
        else:
            return final_feature

    def compute_conv_feature_map_size(self, input_size):
        output = 0
        output += self.conv_block1.compute_conv_feature_map_size(input_size)
        output += self.conv_block2.compute_conv_feature_map_size(input_size)
        output += self.conv_block3.compute_conv_feature_map_size(input_size)
        return output

class PDConv(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False,
                 has_final_process: bool = True,
                 is_dw_conv: bool = False,
                 kernel_size: int = 3,
                 pdconv=PDConv_A
                 ):
        super().__init__()
        self.pdconv = pdconv(input_channels, output_channels, initial_stride, conv_bias, norm_op, norm_op_kwargs,
                             dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first, has_final_process,
                             is_dw_conv, kernel_size)

    def forward(self, x):
        return self.pdconv(x)

    def compute_conv_feature_map_size(self, input_size):
        output = 0
        output += self.pdconv.compute_conv_feature_map_size(input_size)
        return output


class PDConvBlock(nn.Module):
    def __init__(self,
                 num_convs: int,
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False,
                 has_final_process: bool = True
                 ):
        super().__init__()
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        if isinstance(initial_stride, int):
            initial_stride = [initial_stride for _ in range(3)]

        self.initial_stride = initial_stride

        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            PDConv(
                input_channels, output_channels[0], initial_stride, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first,
                has_final_process=has_final_process
            ),
            *[
                PDConv(
                    output_channels[i - 1], output_channels[i], 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first,
                    has_final_process=has_final_process
                )
                for i in range(1, num_convs)
            ]
        )

        self.output_channels = output_channels[-1]

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(
            self.initial_stride), "just give the image size without color/feature channels or " \
                                  "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                  "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output