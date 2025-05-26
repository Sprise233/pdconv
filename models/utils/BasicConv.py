from typing import Union, Type, List, Tuple

import numpy as np
import torch
from torch.nn.modules.dropout import _DropoutNd
import torch.nn as nn

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list


class BasicConv(nn.Module):
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
                 kernel_size: Union[int, List[int], Tuple[int, ...]] = 3
                 ):
        super().__init__()
        if isinstance(initial_stride, int):
            initial_stride = [initial_stride for _ in range(3)]

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        self.nonlin_first = nonlin_first
        self.has_final_process = has_final_process
        self.dropout_op = dropout_op
        self.conv_bias = conv_bias
        self.initial_stride = initial_stride
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.is_dw_conv = is_dw_conv
        self.kernel_size = maybe_convert_scalar_to_list(nn.Conv3d, kernel_size)
        self.padding = [(i - 1) // 2 for i in self.kernel_size]
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs

        if has_final_process:
            final_process_modules = nn.ModuleList([])
            if dropout_op is not None:
                final_process_modules.append(
                    dropout_op(**dropout_op_kwargs)
                )
            if norm_op is not None:
                final_process_modules.append(
                    norm_op(output_channels, **norm_op_kwargs)
                )
            if nonlin is not None:
                final_process_modules.append(
                    nonlin(**nonlin_kwargs)
                )
            self.final_process = nn.Sequential(
                *final_process_modules
            )


class BasicConv3D(BasicConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.is_dw_conv:
            self.group_num = self.input_channels
        else:
            self.group_num = 1
        self.conv_block = nn.Conv3d(in_channels=self.input_channels, out_channels=self.output_channels,
                                    kernel_size=self.kernel_size, stride=self.initial_stride, padding=self.padding,
                                    bias=self.conv_bias, groups=self.group_num)

    def forward(self, x):
        final_feature = self.conv_block(x)

        if self.has_final_process:
            return self.final_process(final_feature)
        else:
            return final_feature

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(
            self.initial_stride), "just give the image size without color/feature channels or " \
                                  "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                  "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.initial_stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64) if self.group_num is None else np.prod(
            [self.output_channels, *output_size], dtype=np.int64) // self.output_channels

class BasicConvTranspose3D(BasicConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.is_dw_conv:
            self.group_num = self.input_channels
        else:
            self.group_num = 1
        self.conv_block = nn.ConvTranspose3d(in_channels=self.input_channels, out_channels=self.output_channels,
                                    kernel_size=self.kernel_size, stride=self.initial_stride, padding=self.padding,
                                    bias=self.conv_bias, groups=self.group_num)

    def forward(self, x):
        final_feature = self.conv_block(x)

        if self.has_final_process:
            return self.final_process(final_feature)
        else:
            return final_feature

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(
            self.initial_stride), "just give the image size without color/feature channels or " \
                                  "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                  "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.initial_stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64) if self.group_num is None else np.prod(
            [self.output_channels, *output_size], dtype=np.int64) // self.output_channels
