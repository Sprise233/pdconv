import numpy as np
from copy import deepcopy
from typing import Union, List, Tuple

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from torch import nn

from models.PDCN.architecture import PDCN
from nnunetv2.experiment_planning.experiment_planners.BasicPDCNPlanner import BasicPDCNPlanner
from nnunetv2.experiment_planning.experiment_planners.basic_experiment_planner import BasicPlanner
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner

from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props


class PDCNPlanner(BasicPDCNPlanner):
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
        self.UNet_blocks_per_stage_encoder = (2, 1, 1, 1, 1, 1)
        self.UNet_blocks_per_stage_decoder = (2, 2, 2, 2, 2)

        self.is_3d = True
        self.is_2d = False

        self.spacing_appoint = spacing
        self.patch_size_appoint = patch_size_appoint

        self.network_class_name = "models.PDCN.architecture.PDCN"

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


if __name__ == '__main__':
    print(PDCNPlanner(dataset_name_or_id=4).get_plans_for_configuration(spacing=(3, 0.78125, 0.78125),
                                    median_shape=(108, 512,512),
                                    data_identifier="PDCNWithPDCDWithPDFMPlans_3d_lowres", approximate_n_voxels_dataset=1.0))