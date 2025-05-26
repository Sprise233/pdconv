import multiprocessing
import shutil
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def copy_BraTS_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str) -> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 4]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


def convert_brats2021(input_folder, d):
    brats_data_dir = input_folder

    task_id = d
    task_name = "BraTS2021"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    case_ids = subdirs(brats_data_dir, prefix='BraTS', join=False)

    for c in case_ids:
        shutil.copy(join(brats_data_dir, c, c + "_t1.nii.gz"), join(imagestr, c + '_0000.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "_t1ce.nii.gz"), join(imagestr, c + '_0001.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "_t2.nii.gz"), join(imagestr, c + '_0002.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "_flair.nii.gz"), join(imagestr, c + '_0003.nii.gz'))

        copy_BraTS_segmentation_and_convert_labels_to_nnUNet(join(brats_data_dir, c, c + "_seg.nii.gz"),
                                                             join(labelstr, c + '.nii.gz'))

    generate_dataset_json(out_base,
                          channel_names={0: 'T1', 1: 'T1ce', 2: 'T2', 3: 'Flair'},
                          labels={
                              'background': 0,
                              'whole tumor': (1, 2, 3),
                              'tumor core': (2, 3),
                              'enhancing tumor': (3,)
                          },
                          num_training_cases=len(case_ids),
                          file_ending='.nii.gz',
                          regions_class_order=(1, 2, 3),
                          license='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          reference='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          dataset_release='1.0')

if __name__ == '__main__':
    brats_data_dir = '/home/isensee/drives/E132-Rohdaten/BraTS_2021/training'

    task_id = 137
    task_name = "BraTS2021"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    case_ids = subdirs(brats_data_dir, prefix='BraTS', join=False)

    for c in case_ids:
        shutil.copy(join(brats_data_dir, c, c + "_t1.nii.gz"), join(imagestr, c + '_0000.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "_t1ce.nii.gz"), join(imagestr, c + '_0001.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "_t2.nii.gz"), join(imagestr, c + '_0002.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "_flair.nii.gz"), join(imagestr, c + '_0003.nii.gz'))

        copy_BraTS_segmentation_and_convert_labels_to_nnUNet(join(brats_data_dir, c, c + "_seg.nii.gz"),
                                                             join(labelstr, c + '.nii.gz'))

    generate_dataset_json(out_base,
                          channel_names={0: 'T1', 1: 'T1ce', 2: 'T2', 3: 'Flair'},
                          labels={
                              'background': 0,
                              'whole tumor': (1, 2, 3),
                              'tumor core': (2, 3),
                              'enhancing tumor': (3, )
                          },
                          num_training_cases=len(case_ids),
                          file_ending='.nii.gz',
                          regions_class_order=(1, 2, 3),
                          license='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          reference='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          dataset_release='1.0')
