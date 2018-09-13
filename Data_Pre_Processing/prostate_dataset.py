#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:06:00 2018

@author: wenyuan

Dataset object for prostate pathological images
"""
import numpy as np
from PIL import Image
import os
import scipy.io
import skimage.io as io



class ProstateDataset():
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        
    def read_image(self, image_id, dir_name = '/tiles', patch_num = -1):
        if patch_num != -1:
            padding = '_' + str(patch_num).zfill(4)
        else:
             padding = ''
        
        img_filename = str(image_id).zfill(4) + padding + '.jpg'
        img_path = os.path.join(self.dataset_dir + dir_name, img_filename)
        img_org = np.array(Image.open(img_path))
        return img_org
    
    def read_original_ann(self, image_id, dir_name = '/masks', patch_num = -1): 
        image_id = str(image_id)
        if patch_num != -1:
            mat_filename = image_id.zfill(4) + '_' + str(patch_num).zfill(4) + '_sementic.mat'
        else:
            mat_filename = image_id.zfill(4) + '.mat'
        mat_file = os.path.join(self.dataset_dir + dir_name, mat_filename)
        key = 'ATmask'
        mat = scipy.io.loadmat(mat_file, mat_dtype=True, 
                               squeeze_me=True, struct_as_record=False)
        return mat[key];
    
    def read_instance_ann(self, image_id, dir_name = '/masks_instance_mod', mode = -1, patch = -1):
        image_id = str(image_id)
        if mode == -1:
            mat_filename = image_id.zfill(4) + '_instance.mat'
        else:
            dir_name = '/masks_instance_mod_' + str(mode)
            mat_filename = image_id.zfill(4) + '_' + str(patch).zfill(4) + '_instance.mat'
        mat_file = os.path.join(self.dataset_dir + dir_name, mat_filename)
        mat = scipy.io.loadmat(mat_file, mat_dtype=True, 
                               squeeze_me=True, struct_as_record=False)
        mask = mat['segmentation']
        class_ids = mat['class_ids']
        return mask, class_ids;
    
    def read_sementic_mod(self, image_id):
        """ read the modified sementic masks
        """
        image_id = str(image_id)
        mat_filename = image_id.zfill(4) + '_sementic.mat'
        mat_file = os.path.join(self.dataset_dir + '/masks_sementic_mod', mat_filename)
        key = 'ATmask'
        mat = scipy.io.loadmat(mat_file, mat_dtype=True, 
                               squeeze_me=True, struct_as_record=False)
        return mat[key];
    
    
    def convert_mat_annotations_to_png(self, ann):
        """ convert mat annotation file to png image 
        
        Parameters
        ----------
        ann
            original annotation matrix.
        
        """
        label_colours = [(128, 0, 0), (0, 128,0), (0, 0, 128), (128, 128, 0)]
        ann = np.transpose(ann)
        h,w = ann.shape    
        outputs = np.zeros((h,w,3), dtype = np.uint8)
        num_classes = 5
    
        img = Image.new('RGB', (h, w))
        pixels = img.load()
        for j_, j in enumerate(ann):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[j_,k_] = label_colours[int(k-1)]
        outputs = np.array(img)
        return outputs
    
    def convert_mat_ann_w_class_id(self, ann, class_ids):
        label_colours = [(128, 0, 0), (0, 128,0), (0, 0, 128), (128, 128, 0)]
        ann = np.transpose(ann)
        h,w = ann.shape    
        outputs = np.zeros((h,w,3), dtype = np.uint8)
    
        img = Image.new('RGB', (h, w))
        pixels = img.load()
        for j_, j in enumerate(ann):
            for k_, k in enumerate(j):
                if k == 1:
                    pixels[j_,k_] = label_colours[int(class_ids)]
        outputs = np.array(img)
        return outputs
    
    def show_ann_png(self, decode_ann):
        im = Image.fromarray(decode_ann)
        im.show()
    
    def show_instance_ann(self, instance_mask, class_ids):
        for i in range(len(class_ids)):
            self.show_ann_png(self.convert_mat_ann_w_class_id(instance_mask[:,:,i], class_ids[i]))
            
    
    
    
if __name__ == '__main__':
    import utils
    data_path = '/data/wenyuan/Path_R_CNN/Data_Pre_Processing'
    dataset_dir = os.path.join(data_path, 'cedars-224')
    start_id = 144
    end_id = 513
    dataset = ProstateDataset(dataset_dir)
    for i in range(start_id, end_id):
        mat = dataset.read_original_ann(i)
        utils.instance_mask_generator(mat, i)
   mask, class_ids = dataset.read_instance_ann(0)
   print(class_ids)
   mat = dataset.read_original_ann(0)
   dataset.show_ann_png(dataset.convert_mat_annotations_to_png(mat))
   dataset.show_instance_ann(mask, class_ids)