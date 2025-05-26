import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def vis_feature_map(feature_map, n_components=3, save_path=None):
    """
    使用PCA提取特征图的主要成分并进行可视化。

    参数:
    feature_map: numpy array or tensor, 3D (C, H, W)
        多通道的特征图，其中 C 是通道数，H 和 W 是高和宽。
    n_components: int, 可选
        PCA中提取的主要成分数量，默认为 3。
    save_path: str, 可选
        图像保存路径，文件格式可以是 .png, .jpg, .pdf 等。如果为 None，图像只会显示，不会保存。
    """

    feature_map = feature_map.cpu()
    feature_map = np.asarray(feature_map)

    if feature_map.ndim != 3:
        raise ValueError('Feature map must be 3D (C, H, W).')

    # 将 (C, H, W) 变成 (C, H * W)，方便 PCA 处理
    C, H, W = feature_map.shape

    sum_map = np.sum(feature_map, axis=(0, ))  # 计算通道的和

    plt.figure(figsize=(10, 5))  # 调整整体图的大小，宽度为15，高度按行数动态调整
    if C <= 1:
        plt.subplot(1, 2, 1)
        plt.imshow(feature_map[0], cmap='viridis')
        plt.colorbar()
        plt.title(f'Principal Component 1')
        plt.subplot(1, 2, 2)
        plt.imshow(sum_map, cmap='viridis')
        plt.colorbar()
        plt.title(f'Principal Component sum')
        if save_path:
            plt.savefig(save_path)
            print(f'feature_map_has_been_saved_in_{save_path}')
        return

    reshaped_map = feature_map.reshape(C, -1)

    # 使用PCA降维，提取 n_components 个主成分
    # 假设 reshaped_map 是输入的张量，形状为 (C, H, W)
    reshaped_map = reshaped_map.reshape(reshaped_map.shape[0], -1)  # 转为 (C, H*W)

    # 计算协方差矩阵的特征值和特征向量
    cov_matrix = np.cov(reshaped_map)
    eig_values, eig_vectors = np.linalg.eigh(cov_matrix)

    # 按特征值降序排列
    sorted_indices = np.argsort(eig_values)[::-1]
    eig_values = eig_values[sorted_indices]
    eig_vectors = eig_vectors[:, sorted_indices]

    # 选择前 n_components 个主成分
    n_components = n_components  # 你需要的主成分数量
    principal_components = eig_vectors[:, :n_components]

    # 投影到主成分空间
    pca_result = np.dot(principal_components.T, reshaped_map)  # (n_components, H*W)
    pca_result = pca_result.reshape(n_components, H, W)  # 恢复形状

    # 设置合理的布局，显示 n_components 个主成分
    num_rows = (n_components + 1) // 3 + (1 if (n_components + 1) % 3 != 0 else 0)  # 动态行数
    plt.figure(figsize=(15, 5 * num_rows))  # 调整整体图的大小，宽度为15，高度按行数动态调整

    for i in range(n_components):
        plt.subplot(num_rows, 3, i + 1)
        plt.imshow(pca_result[i], cmap='viridis')
        plt.colorbar()
        plt.title(f'Principal Component {i + 1}')

    # 在最后一个位置放入 sum_map
    plt.subplot(num_rows, 3, n_components + 1)
    plt.imshow(sum_map, cmap='viridis')
    plt.colorbar()
    plt.title('Principal Component Sum')

    plt.suptitle(f'PCA Result (Top {n_components} Components)')

    # 保存图像到文件
    if save_path:
        plt.savefig(save_path)
        print(f'feature_map_has_been_saved_in_{save_path}')

    plt.show()


