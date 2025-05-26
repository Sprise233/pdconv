import os
import shutil

import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
from surface_distance import metrics
from tqdm import tqdm

from inference.predict_raw import predict_nii_raw_from_imageTs
from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from utils.evaluation import compute_dice_coefficient, compute_hd95
from utils.path import get_result_path
from utils.utils import load_json, load_yaml

from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json

from utils.vis_result import vis_result


def nnUnet_evaluate(config, predict=False, trainer=None):
    label_folder = join(nnUNet_raw, maybe_convert_to_dataset_name(config['dataset']['dataset_name_or_id']), 'labelsTs')
    print(label_folder)
    # 更新预测内容，如果有就不更新
    if not os.path.exists(join(get_result_path(config), f'fold_{config["dataset"]["fold"]}', 'imagesTs_predicted')) or predict:
        # 构造目标文件夹路径
        folder_path = os.path.join(get_result_path(config), f'fold_{config["dataset"]["fold"]}', 'imagesTs_predicted')
        # 检查文件夹是否存在
        if os.path.exists(folder_path):
            try:
                # 删除文件夹及其所有内容
                shutil.rmtree(folder_path)
                print(f"成功删除文件夹: {folder_path}")
            except Exception as e:
                print(f"删除文件夹 {folder_path} 时出错: {e}")
        else:
            print(f"文件夹 {folder_path} 不存在，无需删除")
        predict_nii_raw_from_imageTs(config)

    if config['dataset']['dataset_name_or_id'].startswith('Dataset'):
        pass
    else:
        try:
            dataset_name_or_id = int(config['dataset']['dataset_name_or_id'])
        except ValueError:
            raise ValueError(
                f'dataset_name_or_id must either be an integer or a valid dataset name with the pattern '
                f'DatasetXXX_YYY where XXX are the three(!) task ID digits. Your '
                f'input: {config["dataset"]["dataset_name_or_id"]}')

    result_folder_base = get_result_path(config)

    predicted_folder = join(result_folder_base, f'fold_{config["dataset"]["fold"]}', 'imagesTs_predicted')
    dataset_json = load_json(join(result_folder_base, 'dataset.json'))

    labels_types =  list(dataset_json['labels'].values())[1:]
    dice_scores = {label: [] for label in labels_types}
    hd95_scores = {label: [] for label in labels_types}
    sdc_scores = {label: [] for label in labels_types}


    predicted_file_list  = []

    for index, nii in enumerate(sorted(os.listdir(predicted_folder))):
        if nii.split('.')[-2] + '.' + nii.split('.')[-1] != 'nii.gz':
            continue
        predicted_file_list.append(nii)

    bar = tqdm(total=len(predicted_file_list), desc='测试进度')


    for index, nii in enumerate(sorted(predicted_file_list)):

        truth_image = sitk.ReadImage(os.path.join(label_folder, nii.split('.')[0] + '.nii.gz'))

        # 将 SimpleITK 图像转换为 NumPy 数组
        truth_array = sitk.GetArrayFromImage(truth_image)
        prediction_image = sitk.ReadImage(os.path.join(predicted_folder, nii))
        prediction_array = sitk.GetArrayFromImage(prediction_image)

        # truth_array[:15, :, :] = 0
        # truth_array[-15:, :, :] = 0
        # prediction_array[:15, :, :] = 0
        # prediction_array[-15:, :, :] = 0

        for label in labels_types:
            # 计算每个类别的 DSC
            dice = compute_dice_coefficient(truth_array, prediction_array, label)
            dice_scores[label].append(dice)

            # 计算每个类别的 95% Hausdorff Distance (HD95)
            hd95_value = compute_hd95(truth_array, prediction_array, label)
            hd95_scores[label].append(hd95_value)
            mask_ref = region_or_label_to_mask(truth_array, label)
            mask_pred = region_or_label_to_mask(prediction_array, label)
            surface_distances = metrics.compute_surface_distances(
                mask_ref, mask_pred, (1,1,1)
            )
            sdc_value = metrics.compute_surface_dice_at_tolerance(
                surface_distances,
                tolerance_mm=1)
            sdc_scores[label].append(sdc_value)

        # 存储 DSC 分数的字典
        dice_scores1 = {label: [] for label in labels_types}

        # 可视化 DSC 分数
        # 遍历每一个切片计算DSC分数并保存
        for slice_idx in range(truth_array.shape[0]):
            truth_slice = truth_array[slice_idx, :, :]
            prediction_slice = prediction_array[slice_idx, :, :]

            for label in labels_types:
                if np.all((truth_slice == label) == 0) and np.all((prediction_slice == label) == 0):
                    dice_scores1[label].append(np.nan)
                    continue

                dice = compute_dice_coefficient(truth_slice, prediction_slice, label)
                dice_scores1[label].append(dice)

        # 创建一个图形和单个轴
        fig, ax = plt.subplots(figsize=(10, 5))

        # 遍历所有标签，在同一张图中绘制它们的曲线
        for label in labels_types:
            ax.plot(dice_scores1[label], marker='o', label=f'{label}')

        # 添加图例
        ax.legend()

        # 设置图形标题和坐标轴标签
        ax.set_title(f'{nii.split(".")[0]} DSC Scores for All Labels')
        ax.set_xlabel('Slice Index')
        ax.set_ylabel('DSC Score')
        ax.set_ylim(0, 1)  # DSC 分数的范围是 0 到 1

        # 保存图像
        plt.savefig(os.path.join(predicted_folder, f'{nii.split(".")[0]}_dsc_scores.png'))


        bar.update(1)

    ##################### 以下是可视化和打印最终测试结果 #########################
    vis_result(labels_types, dice_scores, hd95_scores, sdc_scores, save_dir=predicted_folder, result_title='Test', trainer=trainer)

if __name__ == '__main__':
    config = load_yaml('../config.yaml')
    nnUnet_evaluate(config, label_folder='/home/cuiqi/model/nnUnet/data/Dataset001_MBAS/labelsTs')