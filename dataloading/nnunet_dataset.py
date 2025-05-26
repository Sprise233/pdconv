import os
from typing import List

import numpy as np
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers


class Dataset(nnUNetDataset):
    pass


if __name__ == '__main__':
    # this is a mini test. Todo: We can move this to tests in the future (requires simulated dataset)

    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset003_Liver/3d_lowres'
    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=0) # this should not load the properties!
    # this SHOULD HAVE the properties
    ks = ds['liver_0'].keys()
    assert 'properties' in ks
    # amazing. I am the best.

    # this should have the properties
    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=1000)
    # now rename the properties file so that it does not exist anymore
    shutil.move(join(folder, 'liver_0.pkl'), join(folder, 'liver_XXX.pkl'))
    # now we should still be able to access the properties because they have already been loaded
    ks = ds['liver_0'].keys()
    assert 'properties' in ks
    # move file back
    shutil.move(join(folder, 'liver_XXX.pkl'), join(folder, 'liver_0.pkl'))

    # this should not have the properties
    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=0)
    # now rename the properties file so that it does not exist anymore
    shutil.move(join(folder, 'liver_0.pkl'), join(folder, 'liver_XXX.pkl'))
    # now this should crash
    try:
        ks = ds['liver_0'].keys()
        raise RuntimeError('we should not have come here')
    except FileNotFoundError:
        print('all good')
        # move file back
        shutil.move(join(folder, 'liver_XXX.pkl'), join(folder, 'liver_0.pkl'))

