#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:15:09 2018

@author: wenyuan
"""

import utils
from prostate_dataset import ProstateDataset
import numpy as np
import os
 
dataset_dir = os.path.join(os.getcwd(), 'cedars-224')
dataset = ProstateDataset(dataset_dir)

start_id = 0
end_id = 514
display_step = 10

for i in range(start_id, end_id):
    mask, class_ids = dataset.read_instance_ann(i)
    utils.modified_instance_mask(mask, class_ids, i)
    if (i % display_step == 0):
        print("Done modified instance mask prior to %d."%i)

# check if the function goes right

#mask, class_ids = dataset.read_instance_ann(5, dir_name = '/masks_instance_mod')
#dataset.show_instance_ann(mask, class_ids)
