from typing import Tuple, List, Union, Type

import numpy as np
import torch.nn
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from models.utils.helper import maybe_convert_scalar_to_list, is_trans_conv


class ConvDropoutNormReLU(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False,
                 groups=None
                 ):
        super(ConvDropoutNormReLU, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride
        self.groups = groups

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []

        if is_trans_conv(conv_op):
            # 可以根据需要调整padding，这里是个简单示例
            padding = [(i - 1) // 2 for i in kernel_size]
            self.conv = conv_op(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding=padding,
                dilation=1,
                bias=conv_bias,
                groups=groups if groups is not None else 1,
                output_padding=[(s - 1) for s in stride] if kernel_size != stride else 0
            )
        else:
            # 对于普通卷积，保持原来的填充策略
            padding = [(i - 1) // 2 for i in kernel_size]
            self.conv = conv_op(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding=padding,
                dilation=1,
                bias=conv_bias,
                groups=groups if groups is not None else 1
            )

        ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64) if self.groups is None else np.prod(
            [self.output_channels, *output_size], dtype=np.int64) // self.output_channels


class StackedConvBlocks(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False,
                 groups=None
                 ):
        """

        :param conv_op:
        :param num_convs:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        """
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first, groups
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first, groups
                )
                for i in range(1, num_convs)
            ]
        )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

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


# 通用的深度可分离卷积模块，可选择2D或3D卷积
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, input_channels,
                 output_channels,
                 kernel_size: Union[List[int], Tuple[int, ...]],
                 stride: Union[List[int], Tuple[int, ...]],
                 conv_op,
                 norm_op=None,
                 norm_op_kwargs=None,
                 nonlin=None,
                 nonlin_kwargs=None,
                 dw_conv=False
                 ):
        super(DepthwiseSeparableConv, self).__init__()

        # 根据 conv_op 参数选择是 2D 还是 3D 卷积
        self.stride = stride
        self.output_channels = output_channels
        self.dw_conv = dw_conv

        if dw_conv:
            # 深度卷积
            self.depthwise = StackedConvBlocks(num_convs=1,
                                               conv_op=conv_op,
                                               input_channels=input_channels,
                                               output_channels=output_channels,
                                               kernel_size=kernel_size,
                                               initial_stride=stride, groups=input_channels,
                                               norm_op=norm_op,
                                               norm_op_kwargs=norm_op_kwargs,
                                               nonlin=nonlin,
                                               nonlin_kwargs=nonlin_kwargs
                                               )
        else:
            self.depthwise = StackedConvBlocks(num_convs=1,
                                               conv_op=conv_op,
                                               input_channels=input_channels,
                                               output_channels=input_channels,
                                               kernel_size=kernel_size,
                                               initial_stride=stride, groups=input_channels,
                                               norm_op=norm_op,
                                               norm_op_kwargs=norm_op_kwargs,
                                               nonlin=nonlin,
                                               nonlin_kwargs=nonlin_kwargs
                                               )

        conv_op_mapping = {
            nn.ConvTranspose1d: nn.Conv1d,
            nn.ConvTranspose2d: nn.Conv2d,
            nn.ConvTranspose3d: nn.Conv3d
        }

        if not dw_conv:
            # 逐点卷积
            self.pointwise = StackedConvBlocks(num_convs=1,
                                               conv_op=conv_op_mapping.get(conv_op, conv_op),
                                               input_channels=input_channels,
                                               output_channels=output_channels,
                                               kernel_size=(1, 1, 1),
                                               initial_stride=1,
                                               norm_op=norm_op,
                                               norm_op_kwargs=norm_op_kwargs,
                                               nonlin=None,
                                               nonlin_kwargs=None
                                               )

    def forward(self, x):
        # 先进行深度卷积
        x = self.depthwise(x)
        if not self.dw_conv:
            # 然后进行逐点卷积
            x = self.pointwise(x)
        return x

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        if self.dw_conv:
            return np.prod([self.output_channels, *output_size], dtype=np.int64)
        else:
            return np.prod([self.output_channels, *output_size],
                           dtype=np.int64) + self.pointwise.compute_conv_feature_map_size(input_size)


if __name__ == '__main__':
    data = torch.rand((1, 4, 128, 128, 128))

    model = StackedConvBlocks(2, nn.Conv3d, 4, 16, (3, 3, 3), 2,
                              norm_op=nn.InstanceNorm3d, nonlin=nn.ReLU, nonlin_kwargs={'inplace': True},
                              )
    print(model.compute_conv_feature_map_size((128, 128, 128)))
    print(model(data).shape)
    from utils.vis_model import vis_model_from_class

    vis_model_from_class((1, 4, 128, 128, 128), model)
