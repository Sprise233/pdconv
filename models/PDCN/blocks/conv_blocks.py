from typing import Type, Union, List, Tuple

import torch
import torch.nn as nn

from torch.nn.modules.dropout import _DropoutNd

from models.utils.PDConv import PDConv_A as PDConv
from models.utils.helper import maybe_convert_scalar_to_list

from models.utils.simple_conv_blocks import StackedConvBlocks


class EncoderStage(nn.Module):
    def __init__(self,
                 n_blocks: int,
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
                 has_down_sampling: bool = False
                 ):
        super().__init__()

        assert n_blocks > 0, 'n_blocks must be > 0'
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * n_blocks
        if not isinstance(initial_stride, (tuple, list)):
            initial_stride = [initial_stride] * 3

        if not has_down_sampling:
            self.stage_modules = nn.ModuleList([
                PDConv(input_channels, output_channels[0], 1, conv_bias, norm_op,
                       norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
            ])
            for n in range(1, n_blocks):
                self.stage_modules.append(
                    PDConv(output_channels[n - 1], output_channels[n], 1, conv_bias, norm_op,
                           norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs))
        else:
            self.stage_modules = nn.ModuleList([
                DownSampling(input_channels, output_channels[0], initial_stride, conv_bias,
                             norm_op,
                             norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
            ])
            if len(output_channels) >= 1:
                self.stage_modules.append(
                    PDConv(output_channels[0], output_channels[0], 1, conv_bias, norm_op,
                           norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs))
            for n in range(1, n_blocks):
                self.stage_modules.append(
                    PDConv(output_channels[n - 1], output_channels[n], 1, conv_bias, norm_op,
                           norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs))

        self.stage = nn.Sequential(
            *self.stage_modules
        )

        self.initial_stride = maybe_convert_scalar_to_list(nn.Conv3d, initial_stride)

    def forward(self, x):
        return self.stage(x)

    def compute_conv_feature_map_size(self, input_size):
        output = self.stage_modules[0].compute_conv_feature_map_size(input_size)
        input_size = [i // j for i, j in zip(input_size, self.initial_stride)]
        for i, layer in enumerate(self.stage_modules[1:]):
            output += layer.compute_conv_feature_map_size(input_size)

        return output


class DecoderStage(nn.Module):
    def __init__(self,
                 n_blocks: int,
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
                 nonlin_first=False,
                 is_dw_conv: bool = False
                 ):
        super().__init__()

        assert n_blocks > 0, 'n_blocks must be > 0'
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * n_blocks

        self.up_sampling = UpSampling(input_channels, output_channels[0], initial_stride, initial_stride, conv_bias,
                                      norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)

        self.up_sampling_fusion = UpSamplingFusion(output_channels[0], conv_bias, norm_op, norm_op_kwargs, nonlin,
                                                   nonlin_kwargs)

        self.stage_modules = nn.ModuleList([
            PDConv(output_channels[0], output_channels[0], 1, conv_bias, norm_op,
                   norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                   nonlin_first=nonlin_first)
        ])
        for n in range(1, n_blocks):
            self.stage_modules.append(
                PDConv(output_channels[n - 1], output_channels[n], 1, conv_bias, norm_op,
                       norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                       nonlin_first=nonlin_first))

        self.stage = nn.Sequential(
            *self.stage_modules
        )

        self.initial_stride = maybe_convert_scalar_to_list(nn.Conv3d, initial_stride)

    def forward(self, x1, x2):
        return self.stage(self.up_sampling_fusion(self.up_sampling(x1), x2))

    def compute_conv_feature_map_size(self, input_size):
        output = 0
        output += self.stage_modules[0].compute_conv_feature_map_size(input_size)
        for i, layer in enumerate(self.stage_modules[1:]):
            output += layer.compute_conv_feature_map_size(input_size)

        return output


class DownSampling(nn.Module):
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
                 is_dw_conv: bool = False
                 ):
        super().__init__()

        if isinstance(initial_stride, int):
            initial_stride = [initial_stride for _ in range(3)]
        self.initial_stride = initial_stride

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        self.conv_block = StackedConvBlocks(1, nn.Conv3d, input_channels, output_channels, (3, 3, 3),
                                            initial_stride, conv_bias, norm_op,
                                            norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
    def forward(self, x):
        return self.conv_block(x)

    def compute_conv_feature_map_size(self, input_size):
        output = 0
        output += self.conv_block.compute_conv_feature_map_size(input_size)
        return output


class UpSampling(nn.Module):
    def __init__(self,
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
                 ):
        super().__init__()

        if isinstance(initial_stride, int):
            initial_stride = [initial_stride for _ in range(3)]

        self.initial_stride = initial_stride

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size for _ in range(3)]

        self.conv_block = StackedConvBlocks(1, nn.ConvTranspose3d, input_channels, output_channels, kernel_size,
                                            initial_stride, conv_bias, norm_op,
                                            norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)

    def forward(self, x):
        return self.conv_block(x)

    def compute_conv_feature_map_size(self, input_size):
        output = 0
        output += self.conv_block.compute_conv_feature_map_size(input_size)
        return output


class UpSamplingFusion(nn.Module):
    def __init__(self,
                 input_channels: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 ):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2

    def compute_conv_feature_map_size(self, input_size):
        output = 0
        output += self.conv_block.compute_conv_feature_map_size(input_size)
        return output


if __name__ == '__main__':
    data = torch.rand((1, 4, 128, 128, 128))

    model = EncoderStage(2, 4, 16, 3, 2,
                         norm_op=nn.InstanceNorm3d, nonlin=nn.ReLU, nonlin_kwargs={'inplace': True},
                         )
    print(model.compute_conv_feature_map_size((128, 128, 128)))
    from utils.vis_model import vis_model_from_class

    vis_model_from_class((1, 4, 128, 128, 128), model)
