# -*- coding: utf-8 -*-
import os

import argparse

import torch

from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
from nnunetv2.run.run_training import run_training
from nnunetv2.training.nnUNetTrainer.TrainerStatic import TrainerStatic
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from utils.utils import load_yaml

config = load_yaml('config.yaml')
# 设置临时环境变量
os.environ['nnUNet_raw'] = config['environment']['nnUNet_raw']
os.environ['nnUNet_preprocessed'] = config['environment']['nnUNet_preprocessed']
os.environ['nnUNet_results'] = config['environment']['nnUNet_results']

num_process = config['environment']['nnUNet_def_n_proc']

os.environ['nnUNet_def_n_proc'] = num_process
os.environ['nnUNet_n_proc_DA'] = num_process

os.environ["OMP_NUM_THREADS"] = num_process
os.environ["MKL_NUM_THREADS"] = num_process
os.environ["NUMEXPR_NUM_THREADS"] = num_process
os.environ["OPENBLAS_NUM_THREADS"] = num_process
os.environ['VECLIB_MAXIMUM_THREADS'] = num_process
torch.set_num_threads(int(num_process))
torch.set_num_interop_threads(int(num_process))
os.environ["TORCH_COMPILE_DISABLE_FORK"] = num_process

from inference.evaluate import nnUnet_evaluate
from nnunetv2.training.nnUNetTrainer.Trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="处理文件和训练流程")
    parser.add_argument('--save_file_name', type=str, default=config['others']['save_file_name'], help=f"文件名（默认: {config['others']['save_file_name']}）")
    parser.add_argument('--dataset_name_or_id', type=str, default=config['dataset']['dataset_name_or_id'], help=f"数据集（默认: {config['dataset']['dataset_name_or_id']}）")
    parser.add_argument('--fold', type=int, default=config['dataset']['fold'], help=f"数据集（默认: {config['dataset']['fold']}）")
    parser.add_argument('--plans', type=str, default=config['training']['plans'], help=f"数据集（默认: {config['training']['plans']}）")
    parser.add_argument('--num_epochs', type=int, default=config['training']['num_epochs'], help=f"数据集（默认: {config['training']['num_epochs']}）")
    parser.add_argument('--warm_up_epochs', type=int, default=config['training']['warm_up_epochs'], help=f"数据集（默认: {config['training']['warm_up_epochs']}）")
    parser.add_argument('--optimizer_type', type=str, default=config['training']['optimizer_type'], help=f"数据集（默认: {config['training']['optimizer_type']}）")
    parser.add_argument('--initial_lr', type=float, default=config['training']['initial_lr'], help=f"数据集（默认: {config['training']['initial_lr']}）")
    parser.add_argument('--weight_decay', type=float, default=config['training']['weight_decay'], help=f"数据集（默认: {config['training']['weight_decay']}）")
    parser.add_argument('--num_iterations_per_epoch', type=int, default=config['training']['num_iterations_per_epoch'], help=f"数据集（默认: {config['training']['num_iterations_per_epoch']}）")
    parser.add_argument('--num_val_iterations_per_epoch', type=int, default=config['training']['num_val_iterations_per_epoch'], help=f"数据集（默认: {config['training']['num_val_iterations_per_epoch']}）")
    parser.add_argument('--gpu_index', type=int, nargs='+', default=config['training']['gpu_index'], help=f"数据集（默认: {config['training']['gpu_index']}）")
    parser.add_argument('--current_epoch', type=int, default=config['training']['current_epoch'], help=f"数据集（默认: {config['training']['current_epoch']}）")
    parser.add_argument('--only_val', action='store_true', help=f"数据集（默认: {config['validation']['only_validation']}）")
    parser.add_argument('--trainer', type=str, default=config['training']['trainer'],  choices=['Trainer', 'TrainerStatic', 'nnUNetTrainer'], help=f"数据集（默认: {config['training']['trainer']}）")
    parser.add_argument('--do_i_compile', type=bool, default=config['training']['do_i_compile'], help=f"数据集（默认: {config['training']['do_i_compile']}）")


    args = parser.parse_args()

    config['others']['save_file_name'] = args.save_file_name
    config['dataset']['dataset_name_or_id'] = args.dataset_name_or_id
    config['dataset']['fold'] = args.fold
    config['training']['plans'] = args.plans
    config['training']['num_epochs'] = args.num_epochs
    config['training']['warm_up_epochs'] = args.warm_up_epochs
    config['training']['optimizer_type'] = args.optimizer_type
    config['training']['initial_lr'] = args.initial_lr
    config['training']['weight_decay'] = args.weight_decay
    config['training']['num_iterations_per_epoch'] = args.num_iterations_per_epoch
    config['training']['num_val_iterations_per_epoch'] = args.num_val_iterations_per_epoch
    config['training']['current_epoch'] = args.current_epoch
    config['checkpointing']['continue_training'] = args.current_epoch != 0
    config['validation']['only_validation'] = args.only_val
    config['training']['trainer'] = str(args.trainer)
    config['training']['do_i_compile'] = args.do_i_compile

    if isinstance(args.gpu_index, int):
        config['training']['num_gpus'] = 0
        config['training']['gpu_index'] = [args.gpu_index]
    else:
        config['training']['num_gpus'] = len(args.gpu_index)
        config['training']['gpu_index'] = args.gpu_index

    if config['training']['trainer'] == 'Trainer':
        trainer = Trainer(config)
        trainer.run()
    elif config['training']['trainer'] == 'TrainerStatic':
        trainer = TrainerStatic(config)
        trainer.run()
    elif config['training']['trainer'] == 'nnUNetTrainer':
        trainer = None
        run_training(dataset_name_or_id=config['dataset']['dataset_name_or_id'],
                     configuration=config['training']['configuration'],
                     fold=config['dataset']['fold'],
                     trainer_class_name=config['training'].get('trainer_name', 'nnUNetTrainer'),
                     plans_identifier=config['training'].get('plans', 'nnUNetPlans'),
                     use_compressed_data=config['dataset'].get('use_compressed', False),
                     device=torch.device(config['environment']['device']),
                     disable_checkpointing=config['checkpointing'].get('disable_checkpointing', False),
                     continue_training=config['checkpointing'].get('continue_training', False),
                     only_run_validation=config['validation'].get('only_validation', False),
                     pretrained_weights=config['training'].get('pretrained_weights', None),
                     export_validation_probabilities=config['validation'].get('npz', False),
                     val_with_best=config['validation'].get('validate_with_best_checkpoint', False),
                     num_gpus=config['training'].get('num_gpus', 1))

    else:
        trainer = None
    # 测试
    # nnUnet_evaluate(config, predict=True, trainer=trainer)