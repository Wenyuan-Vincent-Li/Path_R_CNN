# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 17:49:54 2017

@author: Wenyuan Li
"""
########## read in the matfile and output png ##############

import numpy as np
from PIL import Image
import os
label_colours = [(128, 0, 0), (0, 128,0), (0, 0, 128), (128, 128, 0)]
                # for mask 1, 2, 3, 4 respectively


def convert_mat_annotations_to_png(masks_root):
    """ Creates a new folder in the root folder of the dataset with annotations stored in .png.
    The function accepts a full path to the root of segmentation
    dataset and converts annotations that are stored in .mat files to .png files. It creates
    a new folder dataset/masks_png where all the converted files will be located. If this
    directory already exists the function does nothing. 
    
    Parameters
    ----------
    masks_root : string
        Full path to the root of patient dataset.
    
    """
    
    import scipy.io
    
    import skimage.io as io
    
    def decode_mat_color(mat_filename, key = 'ATmask'):
    
        mat = scipy.io.loadmat(mat_filename, mat_dtype=True, squeeze_me=True, struct_as_record=False)
        h,w = mat[key].shape

        outputs = np.zeros((h,w,3), dtype = np.uint8)
        num_classes = 5
        
        img = Image.new('RGB', (h, w))
        pixels = img.load()
        for j_, j in enumerate(mat[key]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_,j_] = label_colours[int(k-1)]
        outputs = np.array(img)
        return outputs
    
    def decode_mat_1_channel(mat_filename, key = 'ATmask'):
    
        mat = scipy.io.loadmat(mat_filename, mat_dtype=True, squeeze_me=True, struct_as_record=False)
        
        return mat[key] - 1
    
    mat_file_extension_string = '.mat'
    png_file_extension_string = '.png'
    relative_path_to_annotation_mat_files = 'masks'
    relative_path_to_annotation_png_files = 'masks_1_channel'

    mat_file_extension_string_length = len(mat_file_extension_string)


    annotation_mat_files_fullpath = os.path.join(masks_root,
                                                 relative_path_to_annotation_mat_files)

    annotation_png_save_fullpath = os.path.join(masks_root,
                                                relative_path_to_annotation_png_files)

    # Create the folder where all the converted png files will be placed
    # If the folder already exists, do nothing
    if not os.path.exists(annotation_png_save_fullpath):

        os.makedirs(annotation_png_save_fullpath)
    else:

        return


    mat_files_names = os.listdir(annotation_mat_files_fullpath)

    for current_mat_file_name in mat_files_names:

        current_file_name_without_extention = current_mat_file_name[:-mat_file_extension_string_length]

        current_mat_file_full_path = os.path.join(annotation_mat_files_fullpath,
                                                  current_mat_file_name)

        current_png_file_full_path_to_be_saved = os.path.join(annotation_png_save_fullpath,
                                                              current_file_name_without_extention)
        
        current_png_file_full_path_to_be_saved += png_file_extension_string

#        annotation_array = decode_mat_color(current_mat_file_full_path)
        
        annotation_array = decode_mat_1_channel(current_mat_file_full_path)
        ann = annotation_array.astype(np.uint8)
#        infer_label = Image.fromarray(annotation_array)
        # TODO: hide 'low-contrast' image warning during saving.
        io.imsave(current_png_file_full_path_to_be_saved, ann)

data_path = '/data/wenyuan/Path_R_CNN/Data_Pre_Processing'
data_dir = os.path.join(data_path, 'cedars-224')
convert_mat_annotations_to_png(data_dir)