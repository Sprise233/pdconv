import inspect
import json
import multiprocessing
import os
import subprocess
import sys
import warnings
from asyncio import sleep
from datetime import datetime
from pprint import pprint
from time import time, sleep
from typing import Union, List, Tuple

from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch import distributed as dist
import yaml
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json, save_json, maybe_mkdir_p
from fvcore.nn import FlopCountAnalysis

import nnunetv2
from dataloading.data_loader_2d import DataLoader2D
from dataloading.data_loader_3d import DataLoader3D
from dataloading.nnunet_dataset import Dataset
from dataloading.utils import get_case_identifiers
from loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.configuration import default_num_processes
from nnunetv2.evaluation.evalute_predictions_by_mutil_evaluation import compute_metrics_on_folder
from nnunetv2.evaluation.metrics import ALL_METRICS
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.run_training import find_free_network_port, get_trainer_from_args, maybe_load_checkpoint, \
    setup_ddp, cleanup_ddp
from loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler, PolyLRSchedulerWithWarmup
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels, convert_labelmap_to_one_hot
from utils.path import get_result_path
from utils.utils import load_yaml, get_model_params_count
from torch.cuda import device_count

from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results

from torch.nn.parallel import DistributedDataParallel as DDP

from utils.vis_model import vis_model_from_params


class Trainer(nnUNetTrainer):
    def __init__(self, config: dict):
        """
        训练类。

        参数:
        ----
        config : dict
            一个字典，包含所有的配置参数。
        """

        # initialize nnunetv1 trainer
        self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(
            config['dataset']['dataset_name_or_id']))
        self.plans_file = join(self.preprocessed_dataset_folder_base, config['training']['plans'] + '.json')
        self.plans = load_json(self.plans_file)
        self.dataset_json = load_json(join(self.preprocessed_dataset_folder_base, 'dataset.json'))

        self.config = config

        super(Trainer, self).__init__(plans=self.plans, configuration=config['training']['configuration'],
                                      fold=config['dataset']['fold'],
                                      dataset_json=self.dataset_json,
                                      unpack_dataset=not config['dataset']['use_compressed'],
                                      device=torch.device(config['environment']['device']), config=config)

        ### 使用配置文件中的超参数进行初始化
        self.initial_lr = config['training'].get('initial_lr', 1e-2)
        self.weight_decay = config['training'].get('weight_decay', 3e-5)
        self.oversample_foreground_percent = config['training'].get('oversample_foreground_percent', 0.33)
        self.num_iterations_per_epoch = config['training'].get('num_iterations_per_epoch', 250)
        self.num_val_iterations_per_epoch = config['training'].get('num_val_iterations_per_epoch', 50)
        self.num_epochs = config['training'].get('num_epochs', 20)
        self.current_epoch = config['training'].get('current_epoch', 0)
        self.enable_deep_supervision = config['training'].get('enable_deep_supervision', True)

        # dataloader和dataset
        self.dataloader_2D = DataLoader2D
        self.q = DataLoader3D
        self.dataset = Dataset

        timestamp = datetime.now()
        self.timestamp = timestamp

        if self.config['others']['save_file_name'] is None or self.config['others']['save_file_name'] == '':
            self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))
        else:
            self.log_file = join(self.output_folder, self.config['others']['save_file_name'] + '.txt')

        self.print_to_log_file(f"文件保存地址为：{self.log_file}", also_print_to_console=True)



    def set_unpacking_dataset(self, unpacking_dataset: bool):
        self.unpack_dataset = unpacking_dataset
    def _do_i_compile(self):
        # new default: compile is enabled!
        if torch.__version__.startswith('1.'):
            return False

        if not self.config['training']['do_i_compile']:
            return False

        # compile does not work on mps
        if self.device == torch.device('mps'):
            if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because of unsupported mps device")
            return False

        # CPU compile crashes for 2D models. Not sure if we even want to support CPU compile!? Better disable
        if self.device == torch.device('cpu'):
            if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because device is CPU")
            return False

        # default torch.compile doesn't work on windows because there are apparently no triton wheels for it
        # https://discuss.pytorch.org/t/windows-support-timeline-for-torch-compile/182268/2
        if os.name == 'nt':
            if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because Windows is not natively supported. If "
                                       "you know what you are doing, check https://discuss.pytorch.org/t/windows-support-timeline-for-torch-compile/182268/2")
            return False

        if 'nnUNet_compile' not in os.environ.keys():
            return True
        else:
            return os.environ['nnUNet_compile'].lower() in ('true', '1', 't')

    def print_plans(self):
        if self.local_rank == 0:
            dct = deepcopy(self.plans_manager.plans)
            del dct['configurations']
            self.print_to_log_file(f"\nThis is the configuration used by this "
                                   f"training:\nConfiguration name: {self.configuration_name}\n",
                                   self.configuration_manager, '\n', add_timestamp=False)
            self.print_to_log_file('These are the global plan.json settings:\n', dct, '\n', add_timestamp=False)
            self.print_to_log_file('These are the global config.yaml settings:\n', self.config, '\n',
                                   add_timestamp=False)

    def count_python_processes(self):
        result = subprocess.run("ps -ef | grep '[p]ython'", shell=True, capture_output=True, text=True)
        processes = result.stdout.strip().split('\n')
        processes = [p for p in processes if p]  # 过滤空行

    def initialize(self):

        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            )
            if not (self.config['validation']['only_validation'] or self.config['checkpointing']['continue_training']):
                sample_batch = next(iter(self.dataloader_val))
                vis_model_from_params(get_result_path(self.config),
                                      architecture_class_name=self.configuration_manager.network_arch_class_name,
                                      dummy_input_shape=sample_batch['data'].shape,
                                      arch_init_kwargs=self.configuration_manager.network_arch_init_kwargs,
                                      arch_init_kwargs_req_import=self.configuration_manager.network_arch_init_kwargs_req_import,
                                      num_input_channels=self.num_input_channels,
                                      num_output_channels=self.label_manager.num_segmentation_heads,
                                      enable_deep_supervision=self.enable_deep_supervision)

                # 计算模型参数量
                params_info = get_model_params_count(self.network)
                self.print_to_log_file(f"Total parameters: {params_info['total_params']}")
                self.print_to_log_file(f"Trainable parameters: {params_info['trainable_params']}")
                self.print_to_log_file(f"Non-trainable parameters: {params_info['non_trainable_params']}")

                # self.network.to(self.device)
                # # 使用 fvcore 计算每个模块的 FLOPs
                # flop_analysis = FlopCountAnalysis(self.network, sample_batch['data'].to(self.device))
                # total_flops = flop_analysis.total()
                # total_gflops = total_flops / 1e9
                # self.print_to_log_file(f"Total FLOPs: {total_gflops:.4f} GFLOPs")

            self.network.to(self.device)
            self.count_python_processes()
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)
            self.count_python_processes()

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    @staticmethod
    def print_to_log_file_static(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = (f"{dt_object}:", *args)

            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print(f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)
        elif also_print_to_console:
            print(*args)

    def run(self):
        dataset_name_or_id = self.config['dataset']['dataset_name_or_id']
        configuration = self.config['training']['configuration']
        fold = self.config['dataset']['fold']
        trainer_class_name = self.config['training'].get('trainer_name', 'nnUNetTrainer')
        plans_identifier = self.config['training'].get('plans', 'nnUNetPlans')
        use_compressed_data = self.config['dataset'].get('use_compressed', False)
        device = torch.device(self.config['environment']['device'])
        disable_checkpointing = self.config['checkpointing'].get('disable_checkpointing', False)
        continue_training = self.config['checkpointing'].get('continue_training', False)
        only_run_validation = self.config['validation'].get('only_validation', False)
        pretrained_weights = self.config['training'].get('pretrained_weights', None)
        export_validation_probabilities = self.config['validation'].get('npz', False)
        val_with_best = self.config['validation'].get('validate_with_best_checkpoint', False)
        num_gpus = self.config['training'].get('num_gpus', 1)

        # 将config文件保存到结果文件夹中
        # 将字典保存为 YAML 文件
        config_yaml_save_folder = join(
            get_result_path(self.config),
            f'fold_{fold}',

        )
        # 检查目录是否存在
        if not os.path.exists(config_yaml_save_folder):
            # 如果目录不存在，则创建它
            os.makedirs(config_yaml_save_folder)

        with open(os.path.join(config_yaml_save_folder, 'config.yaml'), 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)

        if plans_identifier == 'nnUNetPlans':
            print("\n############################\n"
                  "INFO: You are using the old nnU-Net default plans. We have updated our recommendations. "
                  "Please consider using those instead! "
                  "Read more here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md"
                  "\n############################\n")

        if isinstance(fold, str):
            if fold != 'all':
                try:
                    fold = int(fold)
                except ValueError as e:
                    print(
                        f'Unable to convert given value for fold to int: {fold}. fold must be either "all" or an integer!')
                    raise e

        if val_with_best:
            assert not disable_checkpointing, '--val_best is not compatible with --disable_checkpointing'

        if num_gpus > 1:
            assert device.type == 'cuda', f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"

            os.environ['MASTER_ADDR'] = 'localhost'
            if 'MASTER_PORT' not in os.environ.keys():
                port = str(find_free_network_port())
                print(f"using port {port}")
                os.environ['MASTER_PORT'] = port  # str(port)

            mp.spawn(self.run_ddp,
                     args=(
                         dataset_name_or_id,
                         configuration,
                         fold,
                         trainer_class_name,
                         plans_identifier,
                         use_compressed_data,
                         disable_checkpointing,
                         continue_training,
                         only_run_validation,
                         pretrained_weights,
                         export_validation_probabilities,
                         val_with_best,
                         num_gpus,
                         self.config['training']['gpu_index']
                    ),
                     nprocs=num_gpus,
                     join=True)
        else:

            if disable_checkpointing:
                self.disable_checkpointing = disable_checkpointing

            assert not (
                    continue_training and only_run_validation), f'Cannot set --c and --val flag at the same time. Dummy.'

            maybe_load_checkpoint(self, continue_training, only_run_validation, pretrained_weights)

            if torch.cuda.is_available():
                cudnn.deterministic = False
                cudnn.benchmark = True

            if not only_run_validation:
                self.run_training()

            if val_with_best:
                self.load_checkpoint(os.path.join(self.output_folder, 'checkpoint_best.pth'))
            self.perform_actual_validation(export_validation_probabilities)

    def run_ddp(self, rank, dataset_name_or_id, configuration, fold, tr, p, use_compressed, disable_checkpointing, c, val,
                pretrained_weights, npz, val_with_best, world_size, gpu_index):
        setup_ddp(rank, world_size)
        torch.cuda.set_device(torch.device('cuda', gpu_index[dist.get_rank()]))

        trainer_name = 'Trainer'

        nnunet_trainer = self

        if nnunet_trainer is None:
            raise RuntimeError(f'Could not find requested nnunetv1 trainer {trainer_name} in '
                               f'nnunetv2.training.nnUNetTrainer ('
                               f'{join(nnunetv2.__path__[0], "training", "nnUNetTrainer")}). If it is located somewhere '
                               f'else, please move it there.')

        # handle dataset input. If it's an ID we need to convert to int from string
        if dataset_name_or_id.startswith('Dataset'):
            pass
        else:
            try:
                dataset_name_or_id = int(dataset_name_or_id)
            except ValueError:
                raise ValueError(
                    f'dataset_name_or_id must either be an integer or a valid dataset name with the pattern '
                    f'DatasetXXX_YYY where XXX are the three(!) task ID digits. Your '
                    f'input: {dataset_name_or_id}')
        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        assert not (c and val), f'Cannot set --c and --val flag at the same time. Dummy.'

        maybe_load_checkpoint(nnunet_trainer, c, val, pretrained_weights)

        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        if not val:
            nnunet_trainer.run_training()

        if val_with_best:
            nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
        nnunet_trainer.perform_actual_validation(npz)
        cleanup_ddp()

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        validation_output_folder = join(self.output_folder, 'validation')
        maybe_mkdir_p(validation_output_folder)

        # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
        # the validation keys across the workers.
        _, val_keys = self.do_split()
        if self.is_ddp:
            last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

            val_keys = val_keys[self.local_rank:: dist.get_world_size()]
            # we cannot just have barriers all over the place because the number of keys each GPU receives can be
            # different

        dataset_val = self.dataset(self.preprocessed_dataset_folder, val_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0)

        next_stages = self.configuration_manager.next_stage_names

        if next_stages is not None:
            _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

        results = []

        for i, k in enumerate(dataset_val.keys()):

            self.print_to_log_file(f"predicting {k}")
            data, seg, properties = dataset_val.load_case(k)

            if self.is_cascaded:
                data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                    output_dtype=data.dtype)))
            with warnings.catch_warnings():
                # ignore 'The given NumPy array is not writable' warning
                warnings.simplefilter("ignore")
                data = torch.from_numpy(data)

            self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
            output_filename_truncated = join(validation_output_folder, k)

            prediction = predictor.predict_sliding_window_return_logits(data)
            prediction = prediction.cpu()

            # this needs to go into background processes
            results.append(
                export_prediction_from_logits(
                    prediction, properties, self.configuration_manager, self.plans_manager,
                    self.dataset_json, output_filename_truncated, save_probabilities),
            )
            if next_stages is not None:
                for n in next_stages:
                    next_stage_config_manager = self.plans_manager.get_configuration(n)
                    expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                        next_stage_config_manager.data_identifier)

                    try:
                        # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                        tmp = self.dataset(expected_preprocessed_folder, [k],
                                           num_images_properties_loading_threshold=0)
                        d, s, p = tmp.load_case(k)
                    except FileNotFoundError:
                        self.print_to_log_file(
                            f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                            f"Run the preprocessing for this configuration first!")
                        continue

                    target_shape = d.shape[1:]
                    output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                    output_file = join(output_folder, k + '.npz')

                    # resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
                    #                   self.dataset_json)
                    results.append(
                        resample_and_save(
                            prediction, target_shape, output_file, self.plans_manager,
                            self.configuration_manager,
                            properties,
                            self.dataset_json),

                    )
            # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
            if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                dist.barrier()

        # for r in results:
        #     print(r)
        # _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else default_num_processes,
                                                # voxel_spacing=self.plans['configurations'][self.config['training']['configuration']]['spacing']
                                                voxel_spacing=(1,1,1)
                                                )
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)
            self.print_to_log_file("Mean Validation SDC: ", (metrics['foreground_mean']["SDC"]),
                                   also_print_to_console=True)
            # self.print_to_log_file("Mean Validation HD95: ", (metrics['foreground_mean']["HD95"]),
            #                        also_print_to_console=True)


        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()


    # 计算损失
    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


    # 设置优化器
    def configure_optimizers(self):
        if self.config['training']['optimizer_type'] == "SGD":
            optimizer = torch.optim.SGD(self.network.parameters(),
                                        lr=self.config['training']['initial_lr'],
                                        weight_decay=self.config['training']['weight_decay'],
                                        momentum=0.99,
                                        nesterov=True)
            lr_scheduler = PolyLRSchedulerWithWarmup(optimizer, self.initial_lr, self.num_epochs,
                                                     warmup_steps=self.config['training']['warm_up_epochs'])

        elif self.config['training']['optimizer_type'] == "AdamW":
            optimizer = torch.optim.AdamW(self.network.parameters(),
                                          lr=self.config['training']['initial_lr'],
                                          weight_decay=self.config['training']['weight_decay'],
                                          eps=1e-4)
            lr_scheduler = None
        else:
            raise ValueError(f"Unsupported optimizer type: {self.config['training']['optimizer_type']}")

        # lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

        return optimizer, lr_scheduler


    # def _get_deep_supervision_scales(self):
    #     if self.enable_deep_supervision:
    #         deep_supervision_scales = list(list(i) for i in 2 / np.cumprod(np.vstack(
    #             self.configuration_manager.pool_op_kernel_sizes), axis=0))
    #     else:
    #         deep_supervision_scales = None  # for train and val_transforms
    #     return deep_supervision_scales


if __name__ == '__main__':
    config = load_yaml('../../../config.yaml')

    trainer = Trainer(config)
    trainer.run()
