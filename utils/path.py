from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.paths import nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def get_result_path(config):
    '''
    根据config中的内容生成最终的result保存位置，精确到训练器例如：“Trainer__nnUNetPlans__3d_fullres”
    :param config: 训练配置文件
    :return: result文件路径
    '''

    return  join(
                nnUNet_results,
                maybe_convert_to_dataset_name(config['dataset']['dataset_name_or_id']),
                f'{config["training"]["trainer"]}__{config["training"]["plans"]}__{config["training"]["configuration"]}'
            )

