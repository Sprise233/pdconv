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
import nibabel as nib


def convert_nii_to_nii_gz(input_path, output_path=None):
    """
    将 .nii 文件转换为 .nii.gz 文件
    参数：
        input_path: 输入的 .nii 文件路径
        output_path: 输出的 .nii.gz 文件路径（可选，如果未指定则在原路径生成）
    返回：
        str: 转换后的文件路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} does not exist")

    # 检查输入文件是否为 .nii 格式
    if not input_path.endswith('.nii'):
        raise ValueError("Input file must be in .nii format")

    # 如果未指定输出路径，默认在原路径下生成 .nii.gz 文件
    if output_path is None:
        output_path = input_path + '.gz'

    # 读取 .nii 文件
    try:
        nii_img = nib.load(input_path)
        # 保存为 .nii.gz 格式（会自动压缩）
        nib.save(nii_img, output_path)
        return output_path
    except Exception as e:
        raise Exception(f"Error during conversion: {str(e)}")

def get_all_files(directory):
    """
    获取指定目录下所有文件（包括子文件夹中的文件）
    参数：
        directory: 文件夹路径
    返回：
        list: 包含所有文件绝对路径的列表
    """
    file_list = []

    # 遍历目录树
    for root, dirs, files in os.walk(directory):
        # 对每个文件生成完整路径并添加到列表
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    return file_list

def make_out_dirs(dataset_id: int, task_name="MBAS"):
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
    image_path, mask_path, train_dir, labels_dir, test_dir, out_test_labels_dir, i = args
    # 读取原始图像
    image = sitk.ReadImage(image_path)
    image_output_file = join(train_dir, f'case{i:04}_0000.nii.gz')
    sitk.WriteImage(image, image_output_file)

    # 读取掩膜
    mask = sitk.ReadImage(mask_path)
    label_output_file = join(labels_dir, f'case{i:04}.nii.gz')
    sitk.WriteImage(mask, label_output_file)



def count_files_in_directory(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


def copy_files(src_data_folder: Path, train_dir: Path, labels_dir: Path, test_dir: Path, out_test_labels_dir: Path):
    src_data_folder_train = join(src_data_folder, 'Training', 'original')
    image_path_list = []
    mask_path_list = []
    i = 0
    for src in os.listdir(src_data_folder_train):
        if src.startswith('MBAS_'):
            i += 1
            image_path_list.append(join(src_data_folder_train, f'MBAS_{i:03}', f'MBAS_{i:03}_gt.nii.gz'))
            mask_path_list.append(join(src_data_folder_train, f'MBAS_{i:03}', f'MBAS_{i:03}_label.nii.gz'))

    # 创建参数元组列表
    task_args = [(image_path, mask_path, train_dir, labels_dir, test_dir, out_test_labels_dir, i+1) for i, (image_path, mask_path) in enumerate(zip(image_path_list, mask_path_list))]

    # 创建进程池（根据CPU核心数调整进程数量）
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_case, task_args)

    train_dir_file_count = count_files_in_directory(train_dir)
    labels_dir_file_count = count_files_in_directory(labels_dir)

    assert train_dir_file_count == labels_dir_file_count, f"最终数量错误，训练图像数量为{train_dir_file_count}个，标签为{labels_dir_file_count}个"

    return train_dir_file_count


def convert_MBAS(src_data_folder: str, dataset_id=7):
    out_dir, train_dir, labels_dir, test_dir, out_test_labels_dir = make_out_dirs(dataset_id=dataset_id)
    num_training_cases = copy_files(Path(src_data_folder), train_dir, labels_dir, test_dir, out_test_labels_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "MRI",
        },
        labels={
            "background": 0,
            "A": 1,
            "B": 2,
            "C": 3,
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
    )


