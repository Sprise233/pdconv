import pydoc
import warnings
from typing import Union, List, Tuple

from nnunetv2.training.nnUNetTrainer.Trainer import Trainer
import torch.nn as nn


class TrainerStatic(Trainer):
    def __init__(self, config: dict):
        super().__init__(config)
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        network_class = architecture_class_name

        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        nw_class = pydoc.locate(network_class)
        # sometimes things move around, this makes it so that we can at least recover some of that
        if nw_class is None:
            warnings.warn(f'Network class {network_class} not found. Attempting to locate it within '
                          f'dynamic_network_architectures.architectures...')

            if nw_class is not None:
                print(f'FOUND IT: {nw_class}')
            else:
                raise ImportError('Network class could not be found, please check/correct your plans file')

        return nw_class(
            input_channels = num_input_channels,
            output_channels = num_output_channels,
            **architecture_kwargs
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        return