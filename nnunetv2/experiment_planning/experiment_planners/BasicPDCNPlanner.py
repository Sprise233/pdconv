import numpy as np
from copy import deepcopy
from typing import Union, List, Tuple

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from torch import nn

from models.PDCN.architecture import PDCN
from nnunetv2.experiment_planning.experiment_planners.basic_experiment_planner import BasicPlanner
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner

from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props, \
    get_shape_must_be_divisible_by, pad_shape


class BasicPDCNPlanner(BasicPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'PDCNPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = True, spacing=None, patch_size_appoint=None):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = PDCN
        # the following two numbers are really arbitrary and were set to reproduce default nnU-Net's configurations as
        # much as possible
        self.UNet_reference_val_3d = 1100000000 # 1032304640
        self.UNet_reference_val_2d = 85000000
        self.UNet_blocks_per_stage_encoder = (2, 2, 2, 2, 2, 2)
        self.UNet_blocks_per_stage_decoder = (2, 2, 2, 2, 2)

        self.is_3d = True
        self.is_2d = False

        self.spacing_appoint = spacing
        self.patch_size_appoint = patch_size_appoint

        self.network_class_name = ""

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        if configuration_name == '3d_fullres':
            # we do not deviate from ExperimentPlanner so we can reuse its data
            return 'PDCNPlans' + '_' + configuration_name
        else:
            return self.plans_identifier + '_' + configuration_name

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        def _features_per_stage(num_stages, max_num_features) -> Tuple[int, ...]:
            return tuple([min(max_num_features, self.UNet_base_num_features * 2 ** i) for
                          i in range(num_stages)])

        if self.spacing_appoint is not None:
            spacing = self.spacing_appoint

        # spacing = np.array([1., 1., 1.])
        def _keygen(patch_size, strides):
            return str(patch_size) + '_' + str(strides)

        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"
        num_input_channels = len(self.dataset_json['channel_names'].keys()
                                 if 'channel_names' in self.dataset_json.keys()
                                 else self.dataset_json['modality'].keys())
        max_num_features = self.UNet_max_features_2d if len(spacing) == 2 else self.UNet_max_features_3d
        unet_conv_op = convert_dim_to_conv_op(len(spacing))

        # print(spacing, median_shape, approximate_n_voxels_dataset)
        # find an initial patch size
        # we first use the spacing to get an aspect ratio
        tmp = 1 / np.array(spacing)

        # we then upscale it so that it initially is certainly larger than what we need (rescale to have the same
        # volume as a patch of size 256 ** 3)
        # this may need to be adapted when using absurdly large GPU memory targets. Increasing this now would not be
        # ideal because large initial patch sizes increase computation time because more iterations in the while loop
        # further down may be required.
        if self.patch_size_appoint is not None:
            initial_patch_size = self.patch_size_appoint
        else:
            if len(spacing) == 3:
                initial_patch_size = [round(i) for i in tmp * (128 ** 3 / np.prod(tmp)) ** (1 / 3)]
            elif len(spacing) == 2:
                initial_patch_size = [round(i) for i in tmp * (2048 ** 2 / np.prod(tmp)) ** (1 / 2)]
            else:
                raise RuntimeError()

        # clip initial patch size to median_shape. It makes little sense to have it be larger than that. Note that
        # this is different from how nnU-Net v1 does it!
        # todo patch size can still get too large because we pad the patch size to a multiple of 2**n
        initial_patch_size = np.minimum(initial_patch_size, median_shape[:len(spacing)])

        # use that to get the network topology. Note that this changes the patch_size depending on the number of
        # pooling operations (must be divisible by 2**num_pool in each axis)
        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
        shape_must_be_divisible_by = get_pool_and_conv_props(spacing, initial_patch_size,
                                                             self.UNet_featuremap_min_edge_length,
                                                             999999)
        num_stages = min(len(pool_op_kernel_sizes), 6)

        norm = get_matching_instancenorm(unet_conv_op)
        architecture_kwargs = {
            'network_class_name': self.network_class_name,
            'arch_kwargs': {
                'n_stages': num_stages,
                'features_per_stage': _features_per_stage(num_stages, max_num_features),
                'strides': pool_op_kernel_sizes[:num_stages],
                'n_blocks_per_stage': self.UNet_blocks_per_stage_encoder[:num_stages],
                'n_conv_per_stage_decoder': self.UNet_blocks_per_stage_decoder[:num_stages - 1],
                'conv_bias': True,
                'norm_op': norm.__module__ + '.' + norm.__name__,
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None,
                'dropout_op_kwargs': None,
                'nonlin': 'torch.nn.LeakyReLU',
                'nonlin_kwargs': {
                    "inplace": True
                }
            },
            '_kw_requires_import': ('norm_op', 'dropout_op', 'nonlin'),
        }

        # now estimate vram consumption
        if _keygen(patch_size, pool_op_kernel_sizes) in _cache.keys():
            estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
        else:
            estimate = self.static_estimate_VRAM_usage(patch_size,
                                                       num_input_channels,
                                                       len(self.dataset_json['labels'].keys()),
                                                       architecture_kwargs['network_class_name'],
                                                       architecture_kwargs['arch_kwargs'],
                                                       architecture_kwargs['_kw_requires_import'],
                                                       )
            _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        # how large is the reference for us here (batch size etc)?
        # adapt for our vram target
        reference = (self.UNet_reference_val_2d if len(spacing) == 2 else self.UNet_reference_val_3d) * \
                    (self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB)

        print(reference)

        while estimate > reference:
            # print(patch_size)
            # patch size seems to be too large, so we need to reduce it. Reduce the axis that currently violates the
            # aspect ratio the most (that is the largest relative to median shape)
            axis_to_be_reduced = np.argsort([i / j for i, j in zip(patch_size, median_shape[:len(spacing)])])[-1]

            # we cannot simply reduce that axis by shape_must_be_divisible_by[axis_to_be_reduced] because this
            # may cause us to skip some valid sizes, for example shape_must_be_divisible_by is 64 for a shape of 256.
            # If we subtracted that we would end up with 192, skipping 224 which is also a valid patch size
            # (224 / 2**5 = 7; 7 < 2 * self.UNet_featuremap_min_edge_length(4) so it's valid). So we need to first
            # subtract shape_must_be_divisible_by, then recompute it and then subtract the
            # recomputed shape_must_be_divisible_by. Annoying.
            patch_size = list(patch_size)
            tmp = deepcopy(patch_size)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by = \
                get_pool_and_conv_props(spacing, tmp,
                                        self.UNet_featuremap_min_edge_length,
                                        999999)
            patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

            # now recompute topology
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
                shape_must_be_divisible_by = get_pool_and_conv_props(spacing, patch_size,
                                                                     self.UNet_featuremap_min_edge_length,
                                                                     999999)

            num_stages = len(pool_op_kernel_sizes)
            architecture_kwargs['arch_kwargs'].update({
                'n_stages': num_stages,
                'strides': pool_op_kernel_sizes,
                'features_per_stage': _features_per_stage(num_stages, max_num_features),
                'n_blocks_per_stage': self.UNet_blocks_per_stage_encoder[:num_stages],
                'n_conv_per_stage_decoder': self.UNet_blocks_per_stage_decoder[:num_stages - 1],
            })
            if _keygen(patch_size, pool_op_kernel_sizes) in _cache.keys():
                estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
            else:
                estimate = self.static_estimate_VRAM_usage(
                    patch_size,
                    num_input_channels,
                    len(self.dataset_json['labels'].keys()),
                    architecture_kwargs['network_class_name'],
                    architecture_kwargs['arch_kwargs'],
                    architecture_kwargs['_kw_requires_import'],
                )
                _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = self.determine_segmentation_softmax_export_fn()

        normalization_schemes, mask_is_used_for_norm = \
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()



        plan = {
            'data_identifier': data_identifier,
            'preprocessor_name': self.preprocessor_name,
            'batch_size': 2,
            'patch_size': patch_size,
            'median_image_size_in_voxels': median_shape,
            'spacing': spacing,
            'normalization_schemes': normalization_schemes,
            'use_mask_for_norm': mask_is_used_for_norm,
            'resampling_fn_data': resampling_data.__name__,
            'resampling_fn_seg': resampling_seg.__name__,
            'resampling_fn_data_kwargs': resampling_data_kwargs,
            'resampling_fn_seg_kwargs': resampling_seg_kwargs,
            'resampling_fn_probabilities': resampling_softmax.__name__,
            'resampling_fn_probabilities_kwargs': resampling_softmax_kwargs,
            'architecture': architecture_kwargs
        }
        return plan

def get_pool_and_conv_props(spacing, patch_size, min_feature_map_size, max_numpool):
    """
    this is the same as get_pool_and_conv_props_v2 from old nnunetv1

    :param spacing:
    :param patch_size:
    :param min_feature_map_size: min edge length of feature maps in bottleneck
    :param max_numpool:
    :return:
    """
    # todo review this code
    dim = len(spacing)

    current_spacing = deepcopy(list(spacing))
    current_size = deepcopy(list(patch_size))

    pool_op_kernel_sizes = [[1] * len(spacing)]
    conv_kernel_sizes = []

    num_pool_per_axis = [0] * dim
    kernel_size = [1] * dim

    stage_count = 0  # 添加阶段计数器

    while True:
        # exclude axes that we cannot pool further because of min_feature_map_size constraint
        valid_axes_for_pool = [i for i in range(dim) if current_size[i] >= 2 * min_feature_map_size]
        if len(valid_axes_for_pool) < 1:
            break

        spacings_of_axes = [current_spacing[i] for i in valid_axes_for_pool]

        # find axis that are within factor of 2 within smallest spacing
        min_spacing_of_valid = min(spacings_of_axes)
        valid_axes_for_pool = [i for i in valid_axes_for_pool if current_spacing[i] / min_spacing_of_valid < 2]

        # max_numpool constraint
        valid_axes_for_pool = [i for i in valid_axes_for_pool if num_pool_per_axis[i] < max_numpool]

        if len(valid_axes_for_pool) == 1:
            if current_size[valid_axes_for_pool[0]] >= 3 * min_feature_map_size:
                pass
            else:
                break
        if len(valid_axes_for_pool) < 1:
            break

        # 检查阶段数是否达到6，若是则退出
        if stage_count >= 6:
            break

        # now we need to find kernel sizes
        # kernel sizes are initialized to 1. They are successively set to 3 when their associated axis becomes within
        # factor 2 of min_spacing. Once they are 3 they remain 3
        for d in range(dim):
            if kernel_size[d] == 3:
                continue
            else:
                if current_spacing[d] / min(current_spacing) < 2:
                    kernel_size[d] = 3

        other_axes = [i for i in range(dim) if i not in valid_axes_for_pool]

        pool_kernel_sizes = [0] * dim
        for v in valid_axes_for_pool:
            pool_kernel_sizes[v] = 2
            num_pool_per_axis[v] += 1
            current_spacing[v] *= 2
            current_size[v] = np.ceil(current_size[v] / 2)
        for nv in other_axes:
            pool_kernel_sizes[nv] = 1

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(deepcopy(kernel_size))

        stage_count += 1  # 每次成功添加一个阶段，计数器加1

    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    def _to_tuple(lst):
        return tuple(_to_tuple(i) if isinstance(i, list) else i for i in lst)

    # we need to add one more conv_kernel_size for the bottleneck. We always use 3x3(x3) conv here
    conv_kernel_sizes.append([3]*dim)
    return num_pool_per_axis, _to_tuple(pool_op_kernel_sizes), _to_tuple(conv_kernel_sizes), tuple(patch_size), must_be_divisible_by


if __name__ == '__main__':
    print(BasicPDCNPlanner(dataset_name_or_id=4).get_plans_for_configuration(spacing=(3, 0.78125, 0.78125),
                                    median_shape=(108, 512,512),
                                    data_identifier="PDCNWithPDCDWithPDFMPlans_3d_lowres", approximate_n_voxels_dataset=1.0))