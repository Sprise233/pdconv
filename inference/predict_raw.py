# predict a bunch of files
import os
from typing import Union, Tuple

import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
if torch.__version__.startswith('2.'):
    from torch._dynamo import OptimizedModule

import nnunetv2
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from utils.path import get_result_path


class nnUNetPredictorReWrite(nnUNetPredictor):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True,
                 config: dict = None):
        super().__init__(
            tile_step_size,
            use_gaussian,
            use_mirroring,
            perform_everything_on_device,
            device,
            verbose,
            verbose_preprocessing,
            allow_tqdm
        )
        self.config = config

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []

        trainer_name = self.config['training']['trainer']
        configuration_name = self.config['training']['configuration']

        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'))
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['config']['training']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, f'nnunetv2.training.nnUNetTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')
        # print(configuration_manager.network_arch_init_kwargs)
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        ).to(torch.device(device=self.device))

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if torch.__version__.startswith('2.'):
            if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                    and not isinstance(self.network, OptimizedModule):
                print('Using torch.compile')
                self.network = torch.compile(self.network)
    def get_net(self):
        return self.network

def predict_nii_raw_from_imageTs(config):
    predictor = nnUNetPredictorReWrite(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
        config=config
    )

    result_folder = get_result_path(config)

    checkpoint_name = 'checkpoint_final.pth'

    print('模型参数路径：' + join(result_folder, f'fold_{config["dataset"]["fold"]}', checkpoint_name))

    predictor.initialize_from_trained_model_folder(
        result_folder,
        use_folds=(config["dataset"]["fold"], ),
        checkpoint_name=checkpoint_name,
    )

    print(join(nnUNet_raw, f'{maybe_convert_to_dataset_name(config["dataset"]["dataset_name_or_id"])}/imagesTs'))
    print(join(result_folder, f'fold_{config["dataset"]["fold"]}', 'imagesTs_predicted'))
    predictor.predict_from_files(join(nnUNet_raw, f'{maybe_convert_to_dataset_name(config["dataset"]["dataset_name_or_id"])}/imagesTs'),
                                 join(result_folder, f'fold_{config["dataset"]["fold"]}', 'imagesTs_predicted'),
                                 save_probabilities=False, overwrite=True,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    # # predict a numpy array
    # from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    #
    # img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTr/liver_63_0000.nii.gz')])
    # ret = predictor.predict_single_npy_array(img, props, None, None, False)
    #
    # iterator = predictor.get_data_iterator_from_raw_npy_data([img], None, [props], None, 1)
    # ret = predictor.predict_from_data_iterator(iterator, False, 1)

def predict_raw(config, raw_file: str):
    predictor = nnUNetPredictorReWrite(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cpu', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
        config=config
    )

    result_folder = get_result_path(config)

    checkpoint_name = 'checkpoint_final.pth'
    # checkpoint_name = 'checkpoint_best.pth'

    print('模型参数路径：' + join(result_folder, f'fold_{config["dataset"]["fold"]}', checkpoint_name))

    predictor.initialize_from_trained_model_folder(
        result_folder,
        use_folds=(config["dataset"]["fold"],),
        checkpoint_name=checkpoint_name,
    )

    img, props = SimpleITKIO().read_images([raw_file])

    iterator = predictor.get_data_iterator_from_raw_npy_data([img], None, [props], None, 1)
    ret = predictor.predict_from_data_iterator(iterator, False, 1)

    return ret

if __name__ == '__main__':
    from utils.utils import load_yaml
    config = load_yaml('../config.yaml')
    # predict_nii_raw_from_imageTs(config)
    print(predict_raw(config, r'D:\python_code\nnUnet\data\Dataset002_ACDC\imagesTr\patient002_frame01_0000.nii.gz')[0].shape)

