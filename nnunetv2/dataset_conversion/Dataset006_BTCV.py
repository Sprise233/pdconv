import os
import re
import shutil
from pathlib import Path
from typing import List

from batchgenerators.utilities.file_and_folder_operations import nifti_files, join, maybe_mkdir_p, save_json
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
import numpy as np


def make_out_dirs(dataset_id: int, task_name="BTCV"):
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"

    out_dir = Path(nnUNet_raw.replace('"', "")) / dataset_name
    out_train_dir = out_dir / "imagesTr"
    out_labels_dir = out_dir / "labelsTr"
    out_test_dir = out_dir / "imagesTs"
    out_test_labels_dir = out_dir / "labelsTs"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)
    os.makedirs(out_test_labels_dir, exist_ok=True)

    return out_dir, out_train_dir, out_labels_dir, out_test_dir, out_test_labels_dir


def copy_files(src_data_folder: Path, train_dir: Path, labels_dir: Path, test_dir: Path, out_test_labels_dir: Path):
    # 获取训练集集所有子文件
    train_subfolders = [f.path for f in os.scandir(os.path.join(src_data_folder, 'imagesTr'))]

    i = 0

    for case_path in train_subfolders:
        # 获取文件夹名
        folder_name = os.path.basename(case_path)
        if folder_name.startswith('img'):
            # 使用正则表达式提取数字部分
            match = re.match(r'img(\d+)', folder_name)  # 例如：'case_00126'

            if match:
                # 提取数字部分并转换为整数
                number = int(match.group(1))
                shutil.copy(case_path, os.path.join(train_dir, f'case{number:04}_0000.nii.gz'))
                i += 1

    # 获取训练集集所有子文件
    train_subfolders = [f.path for f in os.scandir(os.path.join(src_data_folder, 'labelsTr'))]

    for case_path in train_subfolders:
        # 获取文件夹名
        folder_name = os.path.basename(case_path)
        if folder_name.startswith('label'):
            # 使用正则表达式提取数字部分
            match = re.match(r'label(\d+)', folder_name)  # 例如：'case_00126'

            if match:
                # 提取数字部分并转换为整数
                number = int(match.group(1))
                shutil.copy(case_path, os.path.join(labels_dir, f'case{number:04}.nii.gz'))

    # 获取训练集集所有子文件
    train_subfolders = [f.path for f in os.scandir(os.path.join(src_data_folder, 'imagesTs'))]

    for case_path in train_subfolders:
        # 获取文件夹名
        folder_name = os.path.basename(case_path)
        if folder_name.startswith('img'):
            # 使用正则表达式提取数字部分
            match = re.match(r'img(\d+)', folder_name)  # 例如：'case_00126'

            if match:
                # 提取数字部分并转换为整数
                number = int(match.group(1))
                shutil.copy(case_path, os.path.join(test_dir, f'case{number:04}_0000.nii.gz'))

    return i


def convert_BTCV(src_data_folder: str, dataset_id=4):
    out_dir, train_dir, labels_dir, test_dir, out_test_labels_dir = make_out_dirs(dataset_id=dataset_id)
    num_training_cases = copy_files(Path(src_data_folder), train_dir, labels_dir, test_dir, out_test_labels_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "CT",
        },
        labels={
            "background": 0,
            "spleen": 1,
            "rkid": 2,
            "lkid": 3,
            "gall": 4,
            "eso": 5,
            "liver": 6,
            "sto": 7,
            "aorta": 8,
            "IVC": 9,
            "veins": 10,
            "pancreas": 11,
            "rad": 12,
            "lad": 13
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
    )


