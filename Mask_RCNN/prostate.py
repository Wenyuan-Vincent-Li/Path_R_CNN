#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:16:13 2018

@author: wenyuan
"""

import os
import time
import numpy as np
import scipy.io
from xlrd import open_workbook


import zipfile
import urllib.request
import shutil

from config import Config
import utils
import model as modellib
import visualize
from visualize import display_images

def Mean_pixel(dataset_dir, held_out_set):
    filename = os.path.join(dataset_dir, '5_fold_train_pixel_mean.mat')
    mat = scipy.io.loadmat(filename, 
                           mat_dtype=True, squeeze_me=True, struct_as_record=False)
    return mat[str(held_out_set)]

class ProstateConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "prostate"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 classes
    
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200
    
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000
    
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
    
    # Use image level tumor classification head or not
    USE_TUMORCLASS = False
    
class ProstateDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def load_prostate(self, dataset_dir, subset_id, mode = -1):
        """Load a subset of the Prostate dataset.
        dataset_dir: The root directory of the prostate dataset.
        """
        # Add classes
        self.add_class("prostate", 1, "LG")
        self.add_class("prostate", 2, "HG")
        self.add_class("prostate", 3, "BN")
        ## todo: change the target directory
        for i in subset_id:
            if mode == -1:
                image_id = str(i)
                image_filename = image_id.zfill(4) + '.jpg'
                annotation_filename = image_id.zfill(4) + '_instance.mat'
                image_files_name = os.path.join(dataset_dir, 'tiles/' + image_filename)
                annotation_files_name = os.path.join(dataset_dir, 
                                                     'masks_instance_mod/' + annotation_filename)
                self.add_image(
                    "prostate", image_id = i,
                    path = image_files_name,
                    annotations = annotation_files_name)
            else:
                image_id = str(i)
                for patch in range(mode):
                    image_filename = image_id.zfill(4) + '_' + str(patch).zfill(4) + '.jpg'
                    annotation_filename = \
                    image_id.zfill(4) + '_' + str(patch).zfill(4) + '_instance.mat'
                    
                    image_files_name = os.path.join(dataset_dir, 'tiles_' +\
                                                    str(mode) + '/'+ image_filename)
                    annotation_files_name = os.path.join(dataset_dir, 
                                                         'masks_instance_mod_' +\
                                                         str(mode) + '/' + annotation_filename)
                    self.add_image(
                        "prostate", image_id = i * mode + patch,
                        path = image_files_name,
                        annotations = annotation_files_name)
                
    
    def load_mask(self, image_id):
        """Load instance masks for shapes of the given image ID.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "prostate":
            return super(ProstateDataset, self).load_mask(image_id)
        
        annotations = self.image_info[image_id]["annotations"]
        mat = scipy.io.loadmat(annotations, 
                               mat_dtype=True, squeeze_me=True, struct_as_record=False)
        mask = mat['segmentation']
        class_ids = mat['class_ids']
        
        try:
            _,_, d = mask.shape
        except ValueError:
            class_ids = np.array([class_ids])
            mask = np.expand_dims(mask, axis=-1)
        
        return mask, class_ids.astype(np.int32)

    
    def generator_patition(self, dataset_dir, held_out_set):
        """Generate the five fold partition dataset list, return train and val
        data index
        """
        excel_path = os.path.join(dataset_dir, '5_fold_partition.xlsx')
        wb = open_workbook(excel_path)
        table = wb.sheet_by_index(0)
        
        train_list = []
        val_list = []
        for i in range(table.ncols):
            col = table.col_values(i) 
            for j in col:
                if (j != ''):
                    if (i != held_out_set):
                        train_list.append(int(j))
                    else:
                        val_list.append(int(j))
        
        return train_list, val_list
    
    def load_prob_map(self, image_id):
        """Load instance masks for shapes of the given image ID.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "prostate":
            print("No such function related to the current dataset!")
            return None
        
        annotations = self.image_info[image_id]["annotations"]
        file_dir = os.path.join(os.path.dirname(os.path.dirname(annotations)), 'probs_map')
        file_name = str(image_info["id"]).zfill(4) + '_sementic_probs.mat'
        filepath = os.path.join(file_dir, file_name)
        mat = scipy.io.loadmat(filepath, 
                               mat_dtype=True, squeeze_me=True, struct_as_record=False)
        probs = mat['prob_mask']
        
        return probs

    

if __name__ == '__main__':
    import random
    dataset_dir = "/Users/wenyuan/Documents/MII/Mask-RCNN/Data_Pre_Processing/cedars-224"
    config = ProstateConfig()
    config.display()
    dataset = ProstateDataset()
    train_list, val_list = dataset.generator_patition(dataset_dir, 0)
    val_list = [1, 2, 3, 4, 5]
    dataset.load_prostate(dataset_dir, val_list)
    dataset.prepare()
    ## load and display
    image_id = random.choice(dataset.image_ids)
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)