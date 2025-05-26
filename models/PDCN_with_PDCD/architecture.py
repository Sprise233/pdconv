from typing import Union, List, Tuple, Type

import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He, init_last_bn_before_add_to_0
from models.PDCN_with_PDCD.blocks.decoder import Decoder
from models.PDCN_with_PDCD.blocks.encoder import Encoder
from models.utils.CSAM_modules import CSAM
from models.utils.helper import convert_conv_op_to_dim
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class PDCN_with_PDCD(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,

                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        self.conv_op = nn.Conv3d
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                    f"resolution stages. here: {n_stages}. " \
                                                    f"n_conv_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = Encoder(input_channels, n_stages, features_per_stage, strides,
                               n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                               dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                               nonlin_first=nonlin_first)

        self.decoder = Decoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, nonlin_first=nonlin_first)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                   "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                   "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


if __name__ == '__main__':
    from utils.utils import get_model_params_count

    data_shape = (1, 1, 32, 32, 32)
    data = torch.randn(data_shape)
    model = get_network_from_plans(arch_class_name='models.PDCN_with_PDCD.architecture.PDCN_with_PDCD',
                                   arch_kwargs={'n_stages': 6, 'features_per_stage': [32, 64, 128, 256, 320, 320],
                                                'strides': [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2],
                                                            [1, 2, 2]], 'n_blocks_per_stage': [2, 2, 2, 2, 2, 2],
                                                'n_conv_per_stage_decoder': [2, 2, 2, 2, 2], 'conv_bias': True,
                                                'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm3d',
                                                'norm_op_kwargs': {'eps': 1e-05, 'affine': True}, 'dropout_op': None,
                                                'dropout_op_kwargs': None, 'nonlin': 'torch.nn.LeakyReLU',
                                                'nonlin_kwargs': {'inplace': True},
                                                'deep_supervision': True},
                                   arch_kwargs_req_import=['norm_op', 'dropout_op', 'nonlin'],
                                   input_channels=1,
                                   output_channels=4,
                                   allow_init=True,
                                   deep_supervision=True)

    print(model.compute_conv_feature_map_size((128, 128, 128)))
    from utils.vis_model import vis_model_from_class

    vis_model_from_class(data, model)

    # model = get_network_from_plans(
    #     arch_class_name='dynamic_network_architectures.architectures.unet.ResidualEncoderUNet',
    #     arch_kwargs={'n_stages': 6, 'features_per_stage': [32, 64, 128, 256, 320, 320],
    #                  'conv_op': 'torch.nn.modules.conv.Conv3d',
    #                  'kernel_sizes': [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    #                  'strides': [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]],
    #                  'n_blocks_per_stage': [1, 3, 4, 6, 6, 6], 'n_conv_per_stage_decoder': [1, 1, 1, 1, 1],
    #                  'conv_bias': True, 'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm3d',
    #                  'norm_op_kwargs': {'eps': 1e-05, 'affine': True}, 'dropout_op': None, 'dropout_op_kwargs': None,
    #                  'nonlin': 'torch.nn.LeakyReLU', 'nonlin_kwargs': {'inplace': True}, 'deep_supervision': True},
    #     arch_kwargs_req_import=['conv_op', 'norm_op', 'dropout_op', 'nonlin'],
    #     input_channels=1,
    #     output_channels=4,
    #     allow_init=True,
    #     deep_supervision=True)
    #
    # print(model.compute_conv_feature_map_size((20, 256, 224)))
    # from utils.vis_model import vis_model_from_class
    # vis_model_from_class(data_shape, model)
