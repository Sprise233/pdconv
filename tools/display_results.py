import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from PIL import Image

def display_images(image_paths, titles, labels, max_cols, cmap='gray', colorbar_label='Custom Label', colorbar_color='blue'):
    """
    显示多张图片，支持从文件路径读取图片，并在顶部显示标题，底部显示带颜色的样例标签。

    参数：
    - image_paths: 图片路径列表，每个元素是一个字符串表示文件路径。
    - titles: 标题列表，与图片路径一一对应，显示在图片顶部。
    - labels: 字典，键为标签名称，值为对应的颜色，显示在图片底部。
    - max_cols: 每行最多显示的图片数量。
    - cmap: 颜色映射，默认为 'gray'。
    - colorbar_label: 颜色条的自定义标签，默认为 'Custom Label'。
    - colorbar_color: 颜色条标签的颜色，默认为 'blue'。

    注意：图片路径和标题的数量必须相同。
    """
    # 检查图片路径和标题数量是否一致
    if len(image_paths) != len(titles):
        raise ValueError("图片路径和标题的数量必须相同。")

    # 读取图片数据
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)  # 使用 PIL 打开图片
            images.append(np.array(img))  # 转换为 NumPy 数组
        except Exception as e:
            print(f"无法读取图片 {path}: {e}")
            return

    # 计算总图片数和所需行数
    n_images = len(images)
    rows = ceil(n_images / max_cols)

    # 创建画布和子图
    fig, axes = plt.subplots(rows, max_cols, figsize=(max_cols * 3, rows * 3))
    axes = axes.flatten()  # 将二维 axes 数组展平为一维，便于索引

    # 获取标签名称和颜色
    label_names = list(labels.keys())  # 标签名称列表
    label_colors = list(labels.values())  # 对应颜色列表

    # 遍历每张图片并显示
    for i, (image, title) in enumerate(zip(images, titles)):
        axe = axes[i]
        if image.ndim == 2:  # 灰度图像
            im = axe.imshow(image, cmap=cmap)
            cbar = fig.colorbar(im, ax=axe)  # 添加颜色条
            cbar.set_label(colorbar_label, color=colorbar_color)  # 设置颜色条标签和颜色
            cbar.ax.yaxis.set_tick_params(color=colorbar_color)  # 设置颜色条刻度颜色
            plt.setp(plt.getp(cbar.ax, 'ylabel'), color=colorbar_color)  # 设置标签文字颜色
        else:  # 彩色图像 (RGB)
            axe.imshow(image)  # 彩色图像无需 cmap
        axe.set_title(title, pad=10)  # 设置顶部标题并调整间距
        axe.axis('off')  # 隐藏坐标轴

        # 显示底部标签（样例标签）
        if i < len(label_names):  # 确保不超过标签数量
            label_text = label_names[i]
            label_color = label_colors[i]
            axe.text(0.5, -0.1, label_text, transform=axe.transAxes,
                     ha='center', va='top', color=label_color, fontsize=12)

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # 调整布局并显示
    plt.tight_layout()
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 示例图片路径
    image_paths = [
        r"D:\桌面\nnUnet.png",
        r"D:\桌面\gt.png",
        r"D:\桌面\PDCN.png",
    ]
    # 顶部标题
    titles = [
        'nnUNet',
        'Ground Truth',
        'PDCN'
    ]
    # 底部标签字典
    labels = {
        'RV': 'red',
        'MLV': 'blue',
        'LVC': 'green'
    }
    max_cols = 3  # 每行最多 3 张图片

    # 调用函数显示图片
    display_images(
        image_paths,
        titles,
        labels,
        max_cols,
        cmap='viridis',              # 自定义颜色映射
        colorbar_label='值',         # 自定义颜色条标签
        colorbar_color='purple'      # 自定义颜色条颜色
    )