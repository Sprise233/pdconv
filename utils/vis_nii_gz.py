import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np


def resample_nii_image(image, new_spacing):
    """
    对图像进行重采样
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # 计算新的尺寸
    new_size = [
        int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / new_spacing[2]))),
    ]

    # 创建重采样对象
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())

    # 执行重采样
    resampled_image = resample.Execute(image)
    return resampled_image

def visualize_nii_image(file_path=None, img=None, new_spacing = None, save_path=None):
    if img is None:
        # 读取nii.gz文件
        img = sitk.ReadImage(file_path)

    if new_spacing is not None:
        img = resample_nii_image(img, new_spacing)

    # 获取图像数据
    img_data = sitk.GetArrayFromImage(img)

    # 打印图像的基本信息
    print(f"Image shape: {img_data.shape}")
    print(f'Spacing is : {img.GetSpacing()}')

    # 获取图像中的唯一值
    unique_values = np.unique(img_data)
    print(f"Unique values in the image: {unique_values}")

    # 获取colormap（灰度图）
    cmap = plt.cm.get_cmap('viridis')  # 可以选择其他颜色映射，如 'viridis' 或 'jet'

    # 选择一个切片进行可视化
    slice_idx = img_data.shape[0] // 2  # 中间切片
    slice_data = img_data[slice_idx, :, :]

    # 创建一个显示图像的窗口
    plt.figure(figsize=(8, 6))
    plt.imshow(slice_data)

    if save_path is not None:
        plt.savefig(save_path)

    # 显示图形
    plt.show()

    # 3D 可视化
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')

    # 获取非零点的坐标
    x, y, z = np.where(img_data > 0)

    # 获取对应的点值
    values = img_data[x, y, z]

    # 根据值设置透明度（值小于等于0的点的透明度为0）
    alpha = np.clip(values / np.max(values), 0, 1)  # 透明度映射到[0, 1]区间

    # 绘制3D散点图，点的颜色根据值设定，透明度根据值大小设定
    ax.scatter(x, y, z, c=values, cmap='viridis', marker='o', alpha=alpha, s=0.5)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # 设置轴标签
    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')
    ax.grid(None)
    ax.view_init(elev=8, azim=-46)

    ax.axis('off')

    # 显示3D图
    plt.show()

def apply_mask_to_image(label_path, image_path):
    # 读取标签图像（包含值为0的部分）
    label_img = sitk.ReadImage(label_path)
    label_data = sitk.GetArrayFromImage(label_img)

    # 读取目标图像
    image_img = sitk.ReadImage(image_path)
    image_data = sitk.GetArrayFromImage(image_img)

    # 创建一个掩码，值为0的部分对应掩码为True
    mask = label_data == 0

    # 使用掩码将目标图像中对应位置的值设置为0
    image_data[mask] = 0

    # 将修改后的数据保存到新图像
    output_img = sitk.GetImageFromArray(image_data)
    output_img.CopyInformation(image_img)  # 复制原图的元数据（如Spacing，Origin等）

    return output_img



# 使用示例
file_path = r"E:\datasets\amos_0001_image.nii.gz"
# file_path = r"E:\datasets\kits19\case_00000\imaging.nii.gz"
img = apply_mask_to_image(r"E:\datasets\kits19\case_00000\segmentation.nii.gz",r"E:\datasets\kits19\case_00000\imaging.nii.gz")
img = None
visualize_nii_image(file_path=file_path, img=img)
