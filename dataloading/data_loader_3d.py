import json

import numpy as np
import torch
import yaml
from batchgenerators.utilities.file_and_folder_operations import join
from matplotlib import pyplot as plt
from threadpoolctl import threadpool_limits

from dataloading.nnunet_dataset import Dataset
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


class DataLoader3D(nnUNetDataLoader3D):
    pass


def load_json(json_path):
    with open(json_path, 'r') as file:
        json_dict = json.load(file)

    return json_dict

def load_yaml(yaml_path):
    # 读取 YAML 文件并转换为字典
    with open(yaml_path, 'r', encoding='utf-8') as file:
        yaml_dict = yaml.safe_load(file)

    return yaml_dict

if __name__ == '__main__':
    folder = r'D:\python_code\nnUnet\data\processed\Dataset001_AMOS2022_postChallenge_task1\LgNetPlans_3d_fullres'
    ds = Dataset(folder, num_images_properties_loading_threshold=0)  # this should not load the properties!
    plans = load_yaml('../config.yaml')
    plans_manager = PlansManager(plans)
    dataset_json = load_json(join('D:/python_code/nnUnet/data/Dataset001_AMOS2022_postChallenge_task1', 'dataset.json'))
    label_manager = plans_manager.get_label_manager(dataset_json)

    dl = nnUNetDataLoader3D(ds, 1, (48,160,160), (48,160,160), label_manager=label_manager, oversample_foreground_percent=0.33)
    a = next(dl)

    # 假设 a 是从 dl 中得到的 tensor，将它转换为 NumPy 数组
    a1 = a['data']  # 如果是 PyTorch tensor，需要转为 NumPy



    # 选择第一个样本来进行可视化
    sample = a1[0, 0, :, :, :]  # 选择 batch 中的第一个样本，并移除通道维度

    # 选择沿 Z 轴的某个切片进行可视化
    z_slice = sample.shape[0] - 1  # 选择中间切片
    slice_data = sample[z_slice, :, :]  # 提取 Z 轴上的切片

    # 可视化该切片
    plt.imshow(slice_data, cmap='gray')
    plt.title(f"Slice at Z = {z_slice}")
    plt.colorbar()
    plt.show()

    # 假设 a 是从 dl 中得到的 tensor，将它转换为 NumPy 数组
    a2 = a['target']  # 如果是 PyTorch tensor，需要转为 NumPy
    print(a2.shape)
    print(np.unique(a2))

    # 选择第一个样本来进行可视化
    sample = a2[0, 0, :, :, :]  # 选择 batch 中的第一个样本，并移除通道维度

    # 选择沿 Z 轴的某个切片进行可视化
    z_slice = sample.shape[0] - 1  # 选择中间切片
    slice_data = sample[z_slice, :, :]  # 提取 Z 轴上的切片

    # 可视化该切片
    plt.imshow(slice_data)
    plt.title(f"Slice at Z = {z_slice}")
    plt.colorbar()
    plt.show()
