import os
import shutil
from pathlib import Path
from typing import List

from batchgenerators.utilities.file_and_folder_operations import nifti_files, join, maybe_mkdir_p, save_json
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
import numpy as np


def make_out_dirs(dataset_id: int, task_name="LAHeart2018"):
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


import os
import SimpleITK as sitk
import nrrd
import nibabel as nib
from tqdm import tqdm


def merge_labels(laendo, lawall):
    """合并两个标签文件，返回合并后的标签"""
    # 假设 laendo 和 lawall 都是 NumPy 数组
    merged = np.zeros_like(laendo, dtype=np.uint8)

    # 将 laendo 中的非0值设置为 1
    merged[laendo > 0] = 1

    # 将 lawall 中的非0值设置为 2
    merged[lawall > 0] = 2

    return merged


def convert_nrrd_to_nii(input_folder, img_dir=None, label_dir=None, label_file_end_with_label=True):
    all_files = []  # 存储所有有效文件的路径
    for subdir, _, files in os.walk(input_folder):
        laendo_path = os.path.join(subdir, 'laendo.nrrd')
        lawall_path = os.path.join(subdir, 'lawall.nrrd')
        lgemri_path = os.path.join(subdir, 'lgemri.nrrd')

        # 检查所需的文件是否存在
        if os.path.exists(laendo_path) and os.path.exists(lawall_path) and os.path.exists(lgemri_path):
            all_files.append((laendo_path, lawall_path, lgemri_path))

    # 使用 tqdm 显示进度条
    for i, (laendo_path, lawall_path, lgemri_path) in tqdm(enumerate(all_files, start=1), total=len(all_files)):
        laendo, _ = nrrd.read(laendo_path)
        lawall, _ = nrrd.read(lawall_path)

        # 合并标签
        merged_labels = merge_labels(laendo, lawall)

        # 读取图像文件
        lgemri, _ = nrrd.read(lgemri_path)

        # 创建 NIfTI 图像
        img_nifti = nib.Nifti1Image(lgemri, np.eye(4))
        label_nifti = nib.Nifti1Image(merged_labels, np.eye(4))

        # 保存为 .nii.gz 文件
        if img_dir is not None:
            nib.save(img_nifti, os.path.join(img_dir, f'LAHeart2018_{i:03d}_0000.nii.gz'))
        if label_dir is not None:
            if label_file_end_with_label:
                nib.save(label_nifti, os.path.join(label_dir, f'LAHeart2018_{i:03d}_label.nii.gz'))
            else:
                nib.save(label_nifti, os.path.join(label_dir, f'LAHeart2018_{i:03d}.nii.gz'))

    return len(all_files)


def convert_laheart2018(src_data_folder: str, dataset_id=5):
    out_dir, train_dir, labels_dir, test_dir, out_test_labels_dir = make_out_dirs(dataset_id=dataset_id)
    num_training_cases = convert_nrrd_to_nii(os.path.join(src_data_folder, 'Training Set'), img_dir=train_dir,
                                             label_dir=labels_dir, label_file_end_with_label=False)
    convert_nrrd_to_nii(os.path.join(src_data_folder, 'Testing Set'), img_dir=test_dir, label_dir=out_test_labels_dir,
                        label_file_end_with_label=True)
    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "cineMRI",
        },
        labels={
            "background": 0,
            "LA": 1,
            "LAW": 2,
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        default='/home/cuiqi/model/nnUnet/data/LAHeart2018',
        type=str,
        help="The downloaded ACDC dataset dir. Should contain extracted 'training' and 'testing' folders.",
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=5, help="nnU-Net Dataset ID, default: 5"
    )
    args = parser.parse_args()
    print("Converting...")
    convert_laheart2018(args.input_folder, args.dataset_id)

    print("Done!")
