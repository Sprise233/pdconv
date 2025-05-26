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

def make_out_dirs(dataset_id: int, task_name="LAScarQS22"):
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
    image_path, mask_path1, mask_path2, train_dir, labels_dir, test_dir, out_test_labels_dir, i = args
    # 读取原始图像
    image = sitk.ReadImage(image_path)
    image_output_file = join(train_dir, f'case{i:04}_0000.nii.gz')
    sitk.WriteImage(image, image_output_file)

    # 读取掩膜
    mask1 = sitk.ReadImage(mask_path1)
    mask2 = sitk.ReadImage(mask_path2)

    # 转换为 NumPy 数组
    array1 = sitk.GetArrayFromImage(mask1)
    array2 = sitk.GetArrayFromImage(mask2)

    # 检查两个数组的形状是否一致
    if array1.shape != array2.shape:
        raise ValueError(
            f"掩膜文件 {mask_path1} ({array1.shape}) 和 {mask_path2} ({array2.shape}) 形状不一致，无法合并！")

    # 创建新的掩膜数组，使用与原始掩膜一致的数据类型（例如 uint16）
    merged_array = np.zeros_like(array1, dtype=np.uint16)  # 假设需要支持更大的值

    # 将 mask_path1 中值为 420 的区域标记为 2
    merged_array[array1 == 420] = 2
    # 将 mask_path2 中值为 1 的区域标记为 1（后赋值覆盖前赋值）
    merged_array[array2 == 1] = 1
    # 重叠区域标记为3
    merged_array[(array1 == 420) & (array2 == 1)] = 3

    # 转换为 SimpleITK 图像
    merged_image = sitk.GetImageFromArray(merged_array)

    # 设置元信息（从原始图像或 mask1 复制）
    merged_image.SetSpacing(image.GetSpacing())
    merged_image.SetOrigin(image.GetOrigin())
    merged_image.SetDirection(image.GetDirection())

    # 保存合并后的掩膜
    label_output_file = join(labels_dir, f'case{i:04}.nii.gz')
    sitk.WriteImage(merged_image, label_output_file)

    # 转换为 NumPy 数组
    # image_array = sitk.GetArrayFromImage(image)

    # # 获取原始形状
    # z, y, x = image_array.shape
    # target_size = 360
    #
    # # 检查原始图像是否足够大
    # if y < target_size or x < target_size:
    #     raise ValueError(
    #         f"图像 {image_path} 的 xy 大小 ({y}, {x}) 小于目标大小 ({target_size}, {target_size})，无法裁剪！")
    #
    # # 计算裁剪的起始和结束位置（居中裁剪）
    # start_y = (y - target_size) // 2
    # end_y = start_y + target_size
    # start_x = (x - target_size) // 2
    # end_x = start_x + target_size
    #
    # # 确保裁剪范围有效（冗余检查）
    # if start_y < 0 or start_x < 0 or end_y > y or end_x > x:
    #     raise ValueError(
    #         f"裁剪范围无效：start_y={start_y}, end_y={end_y}, start_x={start_x}, end_x={end_x}，原始大小=({y}, {x})")
    #
    # # 裁剪图像数组
    # cropped_image_array = image_array[:, start_y:end_y, start_x:end_x]
    #
    # # 将裁剪后的数组转换回 SimpleITK 图像
    # cropped_image = sitk.GetImageFromArray(cropped_image_array)
    #
    # # 获取并设置元信息
    # spacing = image.GetSpacing()
    # origin = image.GetOrigin()
    # direction = image.GetDirection()
    #
    # # 计算新的原点（考虑方向矩阵）
    # direction_matrix = np.array(direction).reshape(3, 3)
    # spacing_array = np.array(spacing)
    # offset = np.array([start_x, start_y, 0]) * spacing_array  # 偏移向量
    # new_origin = np.array(origin) + direction_matrix @ offset  # 使用方向矩阵调整原点
    #
    # cropped_image.SetSpacing(spacing)
    # cropped_image.SetOrigin(tuple(new_origin))  # 转换为元组
    # cropped_image.SetDirection(direction)
    #
    # # 保存裁剪后的图像
    # output_file = join(train_dir, f'case{i:04}_0000.nii.gz')
    # sitk.WriteImage(cropped_image, output_file)
    #
    # 读取两个掩膜文件
    # mask1 = sitk.ReadImage(mask_path1)
    # mask2 = sitk.ReadImage(mask_path2)
    #
    # # 转换为 NumPy 数组
    # array1 = sitk.GetArrayFromImage(mask1)
    # array2 = sitk.GetArrayFromImage(mask2)
    #
    # # 检查两个数组的形状是否一致
    # if array1.shape != array2.shape:
    #     raise ValueError("两个掩膜文件的形状不一致，无法合并！")
    #
    # # 创建新的掩膜数组，初始值为 0
    # merged_array = np.zeros_like(array1, dtype=np.uint8)
    #
    # # 将 mask_path1 中值为 420 的区域标记为 2
    # merged_array[array1 == 420] = 2
    #
    # # 将 mask_path2 中值为 1 的区域标记为 1
    # merged_array[array2 == 1] = 1
    #
    # # 裁剪数组到 xy 轴大小为 360×360，z 轴不变
    # z, y, x = array1.shape  # 原始形状 (z, y, x)
    # target_size = 360
    #
    # # 计算裁剪的起始和结束位置（居中裁剪）
    # start_y = (y - target_size) // 2
    # end_y = start_y + target_size
    # start_x = (x - target_size) // 2
    # end_x = start_x + target_size
    #
    # # 检查原始图像是否足够大
    # if y < target_size or x < target_size:
    #     raise ValueError(f"原始图像的 xy 大小 ({y}, {x}) 小于目标大小 ({target_size}, {target_size})，无法裁剪！")
    #
    # # 裁剪数组
    # cropped_array = merged_array[:, start_y:end_y, start_x:end_x]
    #
    # # 将裁剪后的 NumPy 数组转换回 SimpleITK 图像
    # merged_image = sitk.GetImageFromArray(cropped_array)
    #
    # # 更新裁剪后的空间信息（调整原点和大小）
    # spacing = mask1.GetSpacing()  # 像素间距
    # origin = mask1.GetOrigin()  # 原点
    # direction = mask1.GetDirection()  # 方向
    #
    # # 计算新的原点（由于裁剪了 xy 平面）
    # new_origin = (
    #     origin[0] + start_x * spacing[0],  # x 轴原点偏移
    #     origin[1] + start_y * spacing[1],  # y 轴原点偏移
    #     origin[2]  # z 轴不变
    # )
    #
    # merged_image.SetSpacing(spacing)
    # merged_image.SetOrigin(new_origin)
    # merged_image.SetDirection(direction)

    # 保存为 .nii.gz 文件
    # output_file = join(labels_dir, f'case{i:04}.nii.gz')
    # sitk.WriteImage(image, output_file)


def count_files_in_directory(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


def copy_files(src_data_folder: Path, train_dir: Path, labels_dir: Path, test_dir: Path, out_test_labels_dir: Path):
    src_data_folder_train = join(src_data_folder, 'train_data')
    image_path_list = []
    mask_path_list1 = []
    mask_path_list2 = []
    for i in os.listdir(src_data_folder_train):
        image_path_list.append(join(src_data_folder_train, i, 'enhanced.nii.gz'))
        mask_path_list1.append(join(src_data_folder_train, i, 'atriumSegImgMO.nii.gz'))
        mask_path_list2.append(join(src_data_folder_train, i, 'scarSegImgM.nii.gz'))

    # 创建参数元组列表
    task_args = [(image_path, mask_path1, mask_path2, train_dir, labels_dir, test_dir, out_test_labels_dir, i) for i, (image_path, mask_path1, mask_path2) in enumerate(zip(image_path_list, mask_path_list1, mask_path_list2))]

    # 创建进程池（根据CPU核心数调整进程数量）
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_case, task_args)

    train_dir_file_count = count_files_in_directory(train_dir)
    labels_dir_file_count = count_files_in_directory(labels_dir)

    assert train_dir_file_count == labels_dir_file_count, f"最终数量错误，训练图像数量为{train_dir_file_count}个，标签为{labels_dir_file_count}个"

    return train_dir_file_count


def convert_LAScarQS22_task1(src_data_folder: str, dataset_id=7):
    out_dir, train_dir, labels_dir, test_dir, out_test_labels_dir = make_out_dirs(dataset_id=dataset_id)
    num_training_cases = copy_files(Path(src_data_folder), train_dir, labels_dir, test_dir, out_test_labels_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "MRI",
        },
        labels={
            "background": 0,
            "LA Scar": (1, 3),
            "LA Atrium": (2, 3),
        },
        file_ending=".nii.gz",
        regions_class_order=(1, 2, 3),
        num_training_cases=num_training_cases,
    )


