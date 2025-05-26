import os
import shutil
import numpy as np
from matplotlib import pyplot as plt

# 定义一个函数来同时打印和保存到文件
def print_and_save(text, file):
    print(text)  # 打印到控制台
    file.write(text + '\n')  # 保存到文件并换行

def vis_result(labels_types, dice_scores, hd95_scores, sdc_scores, result_title, save_dir='./results', trainer=None):
    '''
    打印最终的结果并可视化
    Args:
        save_dir:      保存图片的路径
        result_title:   图片的名称
        labels_types:   label类型列表
        dice_scores:    dice评分列表
        hd95_scores:    hd95评分列表
        sdc_scores:     surface dice评分列表

    Returns:    None
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if trainer is None:
        # 打开文件以写入
        with open(f'{save_dir}/test_log.txt', 'w') as file:
            # 计算每个类别的 DSC、HD95 和 SDC 的均值
            dice_means = {label: np.mean(dice_scores[label]) for label in labels_types}
            hd95_means = {label: np.mean(hd95_scores[label]) for label in labels_types}
            sdc_means = {label: np.mean(sdc_scores[label]) for label in labels_types}

            # 输出每个类别的均值 DSC、HD95 和 SDC
            for label in labels_types:
                print_and_save(f'{result_title} Label {label}:', file)
                print_and_save(f'  {result_title} Mean Dice Similarity Coefficient (DSC): {dice_means[label]:.4f}', file)
                print_and_save(f'  {result_title} Mean 95% Hausdorff Distance (HD95): {hd95_means[label]:.4f}', file)
                print_and_save(f'  {result_title} Mean Surface Dice Coefficient (SDC): {sdc_means[label]:.4f}', file)

            # 计算所有类别的总体均值 DSC、HD95 和 SDC
            overall_dice_mean = np.mean([np.mean(dice_scores[label]) for label in labels_types])
            overall_hd95_mean = np.mean([np.mean(hd95_scores[label]) for label in labels_types])
            overall_sdc_mean = np.mean([np.mean(sdc_scores[label]) for label in labels_types])

            print_and_save(f'{result_title} Total Mean Dice Similarity Coefficient (DSC): {overall_dice_mean:.4f}', file)
            print_and_save(f'{result_title} Total Mean 95% Hausdorff Distance (HD95): {overall_hd95_mean:.4f}', file)
            print_and_save(f'{result_title} Total Mean Surface Dice Coefficient (SDC): {overall_sdc_mean:.4f}', file)

            print_and_save('', file)

            # 计算每个类别的 DSC、HD95 和 SDC 的变异系数
            dice_cvs = {label: np.std(dice_scores[label]) / np.mean(dice_scores[label]) if np.mean(dice_scores[label]) != 0 else 0 for label in labels_types}
            hd95_cvs = {label: np.std(hd95_scores[label]) / np.mean(hd95_scores[label]) if np.mean(hd95_scores[label]) != 0 else 0 for label in labels_types}
            sdc_cvs = {label: np.std(sdc_scores[label]) / np.mean(sdc_scores[label]) if np.mean(sdc_scores[label]) != 0 else 0 for label in labels_types}

            # 输出每个类别的 DSC、HD95 和 SDC 的变异系数
            for label in labels_types:
                print_and_save(f'{result_title} Label {label}:', file)
                print_and_save(f'  {result_title} Cvs Dice Similarity Coefficient (DSC): {dice_cvs[label]:.4f}', file)
                print_and_save(f'  {result_title} Cvs 95% Hausdorff Distance (HD95): {hd95_cvs[label]:.4f}', file)
                print_and_save(f'  {result_title} Cvs Surface Dice Coefficient (SDC): {sdc_cvs[label]:.4f}', file)

            print_and_save('', file)

            # 打印每个类别的结果
            for label in labels_types:
                print_and_save(f"{result_title} Results for {label}:", file)
                print_and_save(f"  {result_title} DSC: {dice_scores[label]}", file)
                print_and_save(f"  {result_title} HD95: {hd95_scores[label]}", file)
                print_and_save(f"  {result_title} SDC: {sdc_scores[label]}", file)
                print_and_save('', file)

    else:
        # 计算每个类别的 DSC、HD95 和 SDC 的均值
        dice_means = {label: np.mean(dice_scores[label]) for label in labels_types}
        hd95_means = {label: np.mean(hd95_scores[label]) for label in labels_types}
        sdc_means = {label: np.mean(sdc_scores[label]) for label in labels_types}

        # 输出每个类别的均值 DSC、HD95 和 SDC
        for label in labels_types:
            trainer.print_to_log_file_static(trainer, f'{result_title} Label {label}:')
            trainer.print_to_log_file_static(trainer, f'  {result_title} Mean Dice Similarity Coefficient (DSC): {dice_means[label]:.4f}')
            trainer.print_to_log_file_static(trainer, f'  {result_title} Mean 95% Hausdorff Distance (HD95): {hd95_means[label]:.4f}')
            trainer.print_to_log_file_static(trainer, f'  {result_title} Mean Surface Dice Coefficient (SDC): {sdc_means[label]:.4f}')

        # 计算所有类别的总体均值 DSC、HD95 和 SDC
        overall_dice_mean = np.mean([np.mean(dice_scores[label]) for label in labels_types])
        overall_hd95_mean = np.mean([np.mean(hd95_scores[label]) for label in labels_types])
        overall_sdc_mean = np.mean([np.mean(sdc_scores[label]) for label in labels_types])

        trainer.print_to_log_file_static(trainer, f'{result_title} Total Mean Dice Similarity Coefficient (DSC): {overall_dice_mean:.4f}')
        trainer.print_to_log_file_static(trainer, f'{result_title} Total Mean 95% Hausdorff Distance (HD95): {overall_hd95_mean:.4f}')
        trainer.print_to_log_file_static(trainer, f'{result_title} Total Mean Surface Dice Coefficient (SDC): {overall_sdc_mean:.4f}')

        trainer.print_to_log_file_static(trainer, '')

        # 计算每个类别的 DSC、HD95 和 SDC 的变异系数
        dice_cvs = {label: np.std(dice_scores[label]) / np.mean(dice_scores[label]) if np.mean(dice_scores[label]) != 0 else 0 for label in labels_types}
        hd95_cvs = {label: np.std(hd95_scores[label]) / np.mean(hd95_scores[label]) if np.mean(hd95_scores[label]) != 0 else 0 for label in labels_types}
        sdc_cvs = {label: np.std(sdc_scores[label]) / np.mean(sdc_scores[label]) if np.mean(sdc_scores[label]) != 0 else 0 for label in labels_types}

        # 输出每个类别的 DSC、HD95 和 SDC 的变异系数
        for label in labels_types:
            trainer.print_to_log_file_static(trainer, f'{result_title} Label {label}:')
            trainer.print_to_log_file_static(trainer, f'  {result_title} Cvs Dice Similarity Coefficient (DSC): {dice_cvs[label]:.4f}')
            trainer.print_to_log_file_static(trainer, f'  {result_title} Cvs 95% Hausdorff Distance (HD95): {hd95_cvs[label]:.4f}')
            trainer.print_to_log_file_static(trainer, f'  {result_title} Cvs Surface Dice Coefficient (SDC): {sdc_cvs[label]:.4f}')

        trainer.print_to_log_file_static(trainer, '')

        # 打印每个类别的结果
        for label in labels_types:
            trainer.print_to_log_file_static(trainer, f"{result_title} Results for {label}:")
            trainer.print_to_log_file_static(trainer, f"  {result_title} DSC: {dice_scores[label]}")
            trainer.print_to_log_file_static(trainer, f"  {result_title} HD95: {hd95_scores[label]}")
            trainer.print_to_log_file_static(trainer, f"  {result_title} SDC: {sdc_scores[label]}")

        trainer.print_to_log_file_static(trainer, '')

    # 创建一个图形，包含三个子图（DSC、HD95、SDC）
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 24))

    # 绘制每个类别的 DSC 分数的点图
    for i, label in enumerate(labels_types):
        ax1.plot(np.random.rand(len(dice_scores[label])) * 0.1 + i, dice_scores[label], 'o', label=label)
    ax1.set_xticks(range(len(labels_types)))
    ax1.set_xticklabels(labels_types)
    ax1.set_xlabel('Label')
    ax1.set_ylabel('DSC Score')
    ax1.set_title(f'{result_title} Dot Plot of DSC Scores for Each Label')
    ax1.legend()

    # 绘制每个类别的 HD95 分数的点图
    for i, label in enumerate(labels_types):
        ax2.plot(np.random.rand(len(hd95_scores[label])) * 0.1 + i, hd95_scores[label], 'o', label=label)
    ax2.set_xticks(range(len(labels_types)))
    ax2.set_xticklabels(labels_types)
    ax2.set_xlabel('Label')
    ax2.set_ylabel('HD95 Score')
    ax2.set_title(f'{result_title} Dot Plot of HD95 Scores for Each Label')
    ax2.legend()

    # 绘制每个类别的 SDC 分数的点图
    for i, label in enumerate(labels_types):
        ax3.plot(np.random.rand(len(sdc_scores[label])) * 0.1 + i, sdc_scores[label], 'o', label=label)
    ax3.set_xticks(range(len(labels_types)))
    ax3.set_xticklabels(labels_types)
    ax3.set_xlabel('Label')
    ax3.set_ylabel('SDC Score')
    ax3.set_title(f'{result_title} Dot Plot of SDC Scores for Each Label')
    ax3.legend()

    # 设置全局标题
    fig.suptitle(f'{result_title} Dot Plots of DSC, HD95, and SDC Scores')

    # 调整布局以防止重叠
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存图片
    plt.savefig(os.path.join(save_dir, f'{result_title}_dsc_hd95_sdc_dot_plot.png'))

    # 显示图形
    plt.show()

    # 可选：绘制箱线图（取消注释以启用）
    """
    # 绘制每个类别的 DSC 分数的箱线图
    plt.figure(figsize=(12, 6))
    plt.boxplot([dice_scores[label] for label in labels_types], labels=labels_types)
    plt.xlabel('Label')
    plt.ylabel('DSC Score')
    plt.title('Boxplot of DSC Scores for Each Label')
    plt.savefig(os.path.join(save_dir, f'{result_title}_dsc_boxplot.png'))
    plt.show()

    # 绘制每个类别的 HD95 分数的箱线图
    plt.figure(figsize=(12, 6))
    plt.boxplot([hd95_scores[label] for label in labels_types], labels=labels_types)
    plt.xlabel('Label')
    plt.ylabel('HD95 Score')
    plt.title('Boxplot of HD95 Scores for Each Label')
    plt.savefig(os.path.join(save_dir, f'{result_title}_hd95_boxplot.png'))
    plt.show()

    # 绘制每个类别的 SDC 分数的箱线图
    plt.figure(figsize=(12, 6))
    plt.boxplot([sdc_scores[label] for label in labels_types], labels=labels_types)
    plt.xlabel('Label')
    plt.ylabel('SDC Score')
    plt.title('Boxplot of SDC Scores for Each Label')
    plt.savefig(os.path.join(save_dir, f'{result_title}_sdc_boxplot.png'))
    plt.show()
    """