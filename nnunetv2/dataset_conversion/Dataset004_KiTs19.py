import os
import re
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import List

from batchgenerators.utilities.file_and_folder_operations import nifti_files, join, maybe_mkdir_p, save_json
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
import numpy as np
import SimpleITK as sitk


def flip_image_to_z_axis(image):
    # 通过SimpleITK的坐标轴翻转来调整图像方向
    new_image = sitk.PermuteAxes(image, (2, 1, 0))

    return new_image

def make_out_dirs(dataset_id: int, task_name="KiTs19"):
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


def process_case(args):
    case_path, train_dir, labels_dir, test_dir = args
    try:
        folder_name = os.path.basename(case_path)
        if not folder_name.startswith('case'):
            return

        match = re.match(r'case_(\d+)', folder_name)
        if not match:
            return

        number = int(match.group(1))
        if 0 <= number < 210:
            # 处理训练数据
            img_path = os.path.join(case_path, 'imaging.nii.gz')
            label_path = os.path.join(case_path, 'segmentation.nii.gz')

            img = sitk.ReadImage(img_path)
            label = sitk.ReadImage(label_path)

            img = flip_image_to_z_axis(img)
            label = flip_image_to_z_axis(label)

            train_image_path = os.path.join(train_dir, f'case{number:03}_0000.nii.gz')
            train_label_path = os.path.join(labels_dir, f'case{number:03}.nii.gz')

            sitk.WriteImage(img, train_image_path)
            sitk.WriteImage(label, train_label_path)

        elif 210 <= number < 300:
            # 处理测试数据
            img_path = os.path.join(case_path, 'imaging.nii.gz')

            img = sitk.ReadImage(img_path)
            img = flip_image_to_z_axis(img)

            test_image_path = os.path.join(test_dir, f'case{number:03}_0000.nii.gz')
            sitk.WriteImage(img, test_image_path)

    except Exception as e:
        print(f"Error processing {case_path}: {str(e)}")

def count_files_in_directory(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


def copy_files(src_data_folder: Path, train_dir: Path, labels_dir: Path, test_dir: Path, out_test_labels_dir: Path):
    # 获取所有子文件夹
    subfolders = [f.path for f in os.scandir(src_data_folder) if f.is_dir()]

    # 创建参数元组列表
    task_args = [(case, train_dir, labels_dir, test_dir) for case in subfolders]

    # 创建进程池（根据CPU核心数调整进程数量）
    with Pool(processes=4) as pool:
        pool.map(process_case, task_args)

    train_dir_file_count = count_files_in_directory(train_dir)
    labels_dir_file_count = count_files_in_directory(labels_dir)

    assert train_dir_file_count == labels_dir_file_count, f"最终数量错误，训练图像数量为{train_dir_file_count}个，标签为{labels_dir_file_count}个"

    return train_dir_file_count


def convert_KiTs19(src_data_folder: str, dataset_id=4):
    out_dir, train_dir, labels_dir, test_dir, out_test_labels_dir = make_out_dirs(dataset_id=dataset_id)
    num_training_cases = copy_files(Path(src_data_folder), train_dir, labels_dir, test_dir, out_test_labels_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "CT",
        },
        labels={
            "background": 0,
            "kidney": 1,
            "tumor": 2,
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
    )


