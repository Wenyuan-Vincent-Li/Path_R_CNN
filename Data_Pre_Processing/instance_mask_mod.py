#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:15:09 2018

@author: wenyuan

This coding can be used to modified instance mask files to exclude those with small areas.
To use this code, be sure to check dataset.read_instance_ann to seek out what read files options
you have and check utils.modified_instance_mask to decide the area threshold and save options.
"""

import utils
from prostate_dataset import ProstateDataset
import numpy as np
import os
data_path = '/data/wenyuan/Path_R_CNN/Data_Pre_Processing'
dataset_dir = os.path.join(data_path, 'cedars-224')
dataset = ProstateDataset(dataset_dir)

start_id = 143
end_id = 144
display_step = 10

for i in range(start_id, end_id):
    mask, class_ids = dataset.read_instance_ann(i, mode = 16, patch = 6)
    utils.modified_instance_mask(mask, class_ids, i, th = 10, dir_name = 'cedars-224/')
    ## this will delete the instance that is smaller than a threshold area.
    if (i % display_step == 0):
        print("Done modified instance mask prior to %d."%i)

# check if the function goes right
mask, class_ids = dataset.read_instance_ann(5, dir_name = '/masks_instance_mod')
dataset.show_instance_ann(mask, class_ids)
