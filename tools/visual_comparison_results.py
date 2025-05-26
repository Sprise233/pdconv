import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import SimpleITK as sitk
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm


def dice_coefficient(matrix1, matrix2):
    # 将矩阵转换为布尔类型，确保是二值矩阵
    matrix1 = matrix1.astype(bool)
    matrix2 = matrix2.astype(bool)
    if np.all(matrix1 == 0) and np.all(matrix2 == 0):
        return 0

    intersection = np.logical_and(matrix1, matrix2).sum()
    total_sum = matrix1.sum() + matrix2.sum()

    # 防止分母为0的情况
    if total_sum == 0:
        return 0.0 if intersection == 0 else 0.0

    dice = 2 * intersection / total_sum
    return dice

def display_multiple_nifti_slices(file_paths, file_names, custom_labels, width, height):
    num_files = len(file_paths)
    images = []

    gt = sitk.ReadImage(file_paths[1])
    gt_array = sitk.GetArrayFromImage(gt)

    slices = []
    for i in range(len(gt_array)):
        max_dice = -1
        for j, nii_file_path in enumerate(file_paths[2:]):
            image = sitk.ReadImage(nii_file_path)
            image_array = sitk.GetArrayFromImage(image)
            dice = dice_coefficient(gt_array[i], image_array[i])
            if max_dice <= dice:
                max_dice = dice
                if j == len(file_paths[2:]) - 1  and max_dice > 0:
                    slices.append(i)

    num_slices = len(slices)
    print(slices)
    if num_slices > 10:  # 如果切片数大于10，随机选10个
        selected_slice_indices = sorted(random.sample(slices, 10))
        num_slices = 10
    else:
        if len(gt) >= 10:
            selected_slice_indices = sorted(slices + random.sample((set(range(len(gt_array))) - set(slices)), 10 - num_slices))
            num_slices = 10
        else:
            selected_slice_indices = list(range(num_slices))

    # 读取所有nii.gz文件，并选取中间的10个切片
    for nii_file_path in file_paths:
        image = sitk.ReadImage(nii_file_path)
        image_array = sitk.GetArrayFromImage(image)

        # 确保所有图像使用相同的切片索引
        if selected_slice_indices is not None:
            image_array = image_array[selected_slice_indices, :, :]  # 按索引裁剪切片

        images.append(image_array)



    # 收集标签数据（假设第一个是原图，其余是标签）
    label_arrays = []
    for i in range(1, num_files):
        label_arrays.append(images[i])

    # 获取所有唯一标签值
    unique_labels = set()
    for arr in label_arrays:
        unique_labels.update(np.unique(arr))
    unique_labels = sorted(unique_labels)
    n_labels = len(unique_labels)

    # 创建颜色映射（固定颜色方案）
    if n_labels > 0:
        colors = ['black', 'red', 'lime', 'blue', 'yellow', 'cyan', 'magenta',
                  'orange', 'purple', 'teal', 'olive', 'maroon', 'navy', 'aqua',
                  'fuchsia', 'silver', 'gray', 'skyblue', 'darkgreen', 'coral',
                  'gold', 'pink', 'lavender', 'turquoise', 'khaki', 'plum',
                  'indigo', 'chartreuse', 'beige', 'tomato'][:n_labels]

        cmap_labels = ListedColormap(colors)
        bounds = np.arange(n_labels + 1) - 0.5  # 假设标签是连续整数
        norm_labels = BoundaryNorm(bounds, cmap_labels.N)
    else:
        cmap_labels = None
        norm_labels = None

    # 调整布局增加图例行
    fig = plt.figure(figsize=(5 * num_slices + 2, 5 * num_files + 2))  # 增加图例空间
    gs = gridspec.GridSpec(num_files + 2,  # 增加一行给图例
                           num_slices + 1,
                           figure=fig,
                           width_ratios=[0.5] + [5] * num_slices,
                           height_ratios=[0.5] + [5] * num_files + [1],  # 最后一行给图例
                           wspace=0.027,
                           hspace=0.027)
    # 添加列标题（切片编号）
    for j in range(num_slices):
        ax_col = fig.add_subplot(gs[0, j + 1])
        ax_col.axis('off')
        ax_col.text(0.5, 1, f'Slice {j + 1}',
                    ha='center', va='center', fontsize=40, fontweight='bold')

    # 添加行标签（文件名）
    for i in range(num_files):
        ax_row = fig.add_subplot(gs[i + 1, 0])
        ax_row.axis('off')
        ax_row.text(-1, 0.5, file_names[i],
                    ha='center', va='center',
                    rotation=90, fontsize=40, fontweight='bold')

    # 创建图像显示区域
    axes = []
    for i in range(num_files):
        row_axes = []
        for j in range(num_slices):
            ax = fig.add_subplot(gs[i + 1, j + 1])
            row_axes.append(ax)
        axes.append(row_axes)

    # 显示所有切片图像
    for i, image_array in enumerate(images):
        l, h, w = image_array.shape
        start_h, end_h = h // 2 - height // 2, h // 2 + height // 2
        start_w, end_w = w // 2 - width // 2, w // 2 + width // 2
        for j in range(num_slices):
            # 裁剪切片
            slice_data = image_array[j, start_h:end_h, start_w:end_w]

            if i == 0:  # 原图使用灰度
                axes[i][j].imshow(slice_data, cmap='gray')
            else:  # 标签使用颜色映射
                axes[i][j].imshow(slice_data.astype(int),
                                  cmap=cmap_labels,
                                  norm=norm_labels)
            axes[i][j].axis('off')

    # 添加图例
    if n_labels > 0:
        ax_legend = fig.add_subplot(gs[num_files + 1, :])
        ax_legend.axis('off')

        # 创建图例句柄
        patches = [mpatches.Patch(color=cmap_labels(i), label=custom_labels[i]) for i in range(len(custom_labels))]

        # 设置每行的标签数量，控制换行
        n_cols = min(len(patches), 6)  # 每行最多显示6个标签，可根据需要调整

        # 添加图例（调整字体大小和布局）
        legend = ax_legend.legend(
            handles=patches,
            loc='center',
            ncol=n_cols,
            fontsize=30,
            title='',
            title_fontsize=35,
            frameon=False
        )

        # 调整图例位置，确保位置合理
        legend.set_bbox_to_anchor((0, -0.2, 1, 0.2))

    plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03, wspace=0.3, hspace=0.4)  # 调整底部空间
    plt.draw()
    plt.savefig(f'{dataset_name}.pdf', format='pdf', bbox_inches='tight')
    plt.show()

dataset_config = {
    'acdc': ['background', 'Right Ventricle', 'Myocardium of Left Ventricle', 'Left Ventricle Cavity'],
    'mbas': ['background', 'Left Atrium', 'Right Atrium', 'Atrial Wall'],
    'kits19': ['background', 'Kidney', 'Tumor'],
    'amos22': ['background', 'Liver', 'Right Kidney', 'Spleen', 'Pancreas', 'Aorta', 'Inferior Vena Cava', 'Right Adrenal Gland', 'Left Adrenal Gland', 'Gall Bladder', 'Esophagus', 'Stomach', 'Duodenum', 'Left Kidney', 'Bladder', 'Prostate/Uterus'],
}

dataset_name = 'kits19'
width, height = 300, 300
# 示例用法保持不变
nii_file_paths = [
    f"E:/view/image/{dataset_name}.nii.gz",
    f"E:/view/gt/{dataset_name}.nii.gz",
    f"E:/view/nnUnet/{dataset_name}.nii.gz",
    f"E:/view/UNETR/{dataset_name}.nii.gz",
    f"E:/view/MedNeXt/{dataset_name}.nii.gz",
    f"E:/view/PDC-Net/{dataset_name}.nii.gz",
]

file_names = ['Image', 'Ground Truth', 'nnUnet', 'UNETR', 'MedNeXt', 'PDC-Net']

display_multiple_nifti_slices(nii_file_paths, file_names, dataset_config[dataset_name], width, height)