from typing import Union, List, Tuple, Type

import numpy as np
import torch.nn as nn
import torch
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from models.PDCN_with_PDCD_with_PDFM_dw.blocks.conv_blocks import EncoderStage
from models.utils.helper import maybe_convert_scalar_to_list
from models.utils.simple_conv_blocks import StackedConvBlocks
from utils.vis_attention import vis_feature_map


class Encoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 return_skips: bool = False,
                 disable_default_stem: bool = False,
                 stem_channels: int = None,
                 nonlin_first=False,

                 ):
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if bottleneck_channels is None or isinstance(bottleneck_channels, int):
            bottleneck_channels = [bottleneck_channels] * n_stages
        assert len(
            bottleneck_channels) == n_stages, "bottleneck_channels must be None or have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        strides = [maybe_convert_scalar_to_list(nn.Conv3d, i) for i in strides]
        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]

            self.stem = StackedConvBlocks(
                1,
                nn.Conv3d,
                input_channels, stem_channels,
                (1, 3, 3),
                1,
                conv_bias, norm_op, norm_op_kwargs, dropout_op,
                dropout_op_kwargs, nonlin, nonlin_kwargs,
                nonlin_first=nonlin_first
            )
            input_channels = stem_channels
        else:
            self.stem = None

        # now build the network
        stages = []
        for s in range(n_stages):
            has_down_sampling = True
            if s == 0:
                has_down_sampling = False
            stage = EncoderStage(
                n_blocks_per_stage[s], input_channels, features_per_stage[s],
                strides[s],
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, has_down_sampling=has_down_sampling
            )

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = strides
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.conv_op = nn.Conv3d

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for i, s in enumerate(self.stages):
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output


if __name__ == '__main__':
    data_shape = (1, 4, 128, 128, 128)
    model = Encoder(4, 2, (32, 64), 3, (1, 2), (2), True,
                    nn.BatchNorm3d, None, None, None, nn.ReLU, None, stem_channels=32, return_skips=True,
                    disable_default_stem=False)

    print(model.compute_conv_feature_map_size((128, 128, 128)))
    # # print(model(torch.randn(data_shape)))
    # from utils.vis_model import vis_model_from_class
    #
    # vis_model_from_class(data_shape, model)
