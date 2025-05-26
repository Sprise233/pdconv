import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from threadpoolctl import threadpool_limits

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from utils.utils import load_json, load_yaml


class DataLoader2D(nnUNetDataLoader2D):
    pass


if __name__ == '__main__':
    config = load_yaml('../config.yaml')
    # folder = '/media/fabian/data/nnUNet_preprocessed/Dataset004_Hippocampus/2d'
    folder = '/home/cuiqi/model/nnUnet/processed/Dataset001_MBAS/nnUNetPlans_2d'

    ds = nnUNetDataset(folder, None, 1000)  # this should not load the properties!
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed,
                                            maybe_convert_to_dataset_name(config['dataset']['dataset_name_or_id']))
    plans_file = join(preprocessed_dataset_folder_base, config['training']['plans'] + '.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    label_manager = PlansManager(plans).get_label_manager(dataset_json)
    dl = nnUNetDataLoader2D(ds, 16, (640, 640), (640, 640), label_manager, 0.33, None, None)
    a = next(dl)
