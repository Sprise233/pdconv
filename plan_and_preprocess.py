import argparse

from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
import numpy as np


def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Process datasets and plan experiments.")

    # 参数: 数据集 ID
    parser.add_argument('--dataset_ids', type=int, nargs='+',
                        help="List of dataset IDs to process, e.g., --dataset_ids 1 2 3")

    parser.add_argument('--plan_type', type=str, default='PDCN', choices=['PDCN', 'nnUnet'],
                        help="Plan Type Choice")

    # 参数: 选择的实验类型 (c)
    parser.add_argument('--c', type=str, nargs='+', choices=['2d', '3d_fullres', '3d_lowres'], default=['3d_fullres'],
                        help="List of experiment types, e.g., --c 2d 3d_ ullres")

    # 参数: spacing (间距)
    parser.add_argument('--spacing', type=float, nargs=3, default=None,
                        help="Spacing values as three float numbers, e.g., --spacing 2.5 0.8 0.8")

    # 参数: patch_size (补丁大小)
    parser.add_argument('--patch_size', type=int, nargs=3, default=None,
                        help="Patch size as three integers, e.g., --patch_size 48 224 224")

    # 参数: 默认的 线程数 配置
    parser.add_argument('--default_np', type=int, nargs='+', default=[8, 1, 8],
                        help="Default np configuration for each experiment type, e.g., --default_np 8 16 8")

    parser.add_argument('--gpu_memory', type=int, default=8,
                        help="gpu memory (GB)")

    # 解析命令行参数
    args = parser.parse_args()

    # 获取默认 np 配置
    default_np = {
        "2d": args.default_np[0],
        "3d_fullres": args.default_np[1],
        "3d_lowres": args.default_np[2],
    }

    # 根据实验类型 c 设置 np
    np_values = [default_np[i] if i in default_np.keys() else 4 for i in args.c]

    # 处理提取指纹
    extract_fingerprints(dataset_ids=args.dataset_ids)

    if args.plan_type == 'PDCN':
        # 计划实验
        plans_identifier = plan_experiments(
            dataset_ids=args.dataset_ids,
            gpu_memory_target_in_gb=args.gpu_memory,
            experiment_planner_class_name='PDCNPlanner',
            spacing=np.array(args.spacing) if args.spacing is not None else None,
            patch_size=np.array(args.patch_size) if args.patch_size is not None else None
        )

        # 多次调用 plan_experiments
        plan_experiments(
            dataset_ids=args.dataset_ids,
            gpu_memory_target_in_gb=args.gpu_memory,
            experiment_planner_class_name='PDCNWithPDCDPlanner',
            spacing=np.array(args.spacing) if args.spacing is not None else None,
            patch_size=np.array(args.patch_size) if args.patch_size is not None else None
        )
        plan_experiments(
            dataset_ids=args.dataset_ids,
            gpu_memory_target_in_gb=args.gpu_memory,
            experiment_planner_class_name='PDCNWithPDCDWithPDFMPlanner',
            spacing=np.array(args.spacing) if args.spacing is not None else None,
            patch_size=np.array(args.patch_size) if args.patch_size is not None else None
        )
    elif args.plan_type == 'nnUnet':
        plans_identifier = plan_experiments(
            dataset_ids=args.dataset_ids,
            gpu_memory_target_in_gb=args.gpu_memory,
            experiment_planner_class_name='ExperimentPlanner'
        )
    else:
        plans_identifier = None

    # 预处理
    preprocess(args.dataset_ids, plans_identifier, args.c, np_values, False)


if __name__ == '__main__':
    main()
    # 处理提取指纹
    # extract_fingerprints(dataset_ids=[4])
    # plan_experiments(
    #     dataset_ids=[4],
    #     gpu_memory_target_in_gb=16,
    #     experiment_planner_class_name='ExperimentPlanner'
    # )