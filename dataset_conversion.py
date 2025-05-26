from dataloading.data_loader_3d import load_json
from nnunetv2.dataset_conversion.Dataset004_KiTs19 import convert_KiTs19
from nnunetv2.dataset_conversion.Dataset006_BTCV import convert_BTCV
from nnunetv2.dataset_conversion.Dataset007_LAScarQS2022 import convert_LAScarQS22_task1
from nnunetv2.dataset_conversion.Dataset008_MBAS import convert_MBAS
from nnunetv2.dataset_conversion.Dataset027_ACDC import convert_acdc
from nnunetv2.dataset_conversion.Dataset137_BraTS21 import convert_brats2021
from nnunetv2.dataset_conversion.Dataset218_Amos2022_task1 import convert_amos_task1
from nnunetv2.dataset_conversion.Dataset219_Amos2022_task2 import convert_amos_task2

import argparse

if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Dataset processing script.")
    parser.add_argument('--dataset_type', type=str,
                        choices=['amos2022_task2', 'amos2022_task1', 'acdc', 'brats21', 'kits19', 'btcv', 'LAScarQS22_task1', 'mbas'],
                        help='The dataset type to process.')
    parser.add_argument('--input_folder', type=str,
                        help='The folder where the dataset is located.')
    parser.add_argument('-d', type=int,
                        help='Dataset number.')

    # 解析命令行参数
    args = parser.parse_args()

    # 数据集处理函数映射
    dataset_processors = {
        'amos2022_task2': convert_amos_task2,
        'amos2022_task1': convert_amos_task1,
        'acdc': convert_acdc,
        'brats21': convert_brats2021,
        'kits19': convert_KiTs19,
        'btcv': convert_BTCV,
        'LAScarQS22_task1': convert_LAScarQS22_task1,
        'mbas': convert_MBAS
    }

    # if args.test_list_folder is not None:
    #     test_list = load_json(args.test_list_folder)['test']
    #     for i, test_file_name in enumerate(test_list):
    #         test_list[i] = str(test_file_name).split('/')[-1]
    #     print(test_list)
    # else:
    #     test_list = None


    # 调用对应的函数进行数据集处理
    dataset_processors[args.dataset_type](args.input_folder, args.d)
