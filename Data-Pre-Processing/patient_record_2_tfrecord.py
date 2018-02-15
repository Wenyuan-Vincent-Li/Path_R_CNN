# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from xlrd import open_workbook

from tf_records import write_image_annotation_pairs_to_tfrecord

image_dir = './cedars-224/tiles'
mask_dir = './cedars-224/masks_1_channel'
patient_excel = './cedars-224/Cases_Tiles.xlsx'
wb = open_workbook(patient_excel)
table = wb.sheet_by_index(0)
ten_fold_val = [[13,9], [12,6], [17,5], [11,4], [3,10], [1,19], [7,14], [20,2],
                [16,18], [8,15]]
d = {}
for i in range(table.ncols):
    patient = table.col_values(i)
    name = patient[0]
    value = []
    for j in patient[1 : ]:
        if isinstance(j, float):
            value.append(j)
    d[name] = value

ten_fold_name = []
for i in ten_fold_val:
    filename = []
    for j in i:
        filename += d['P'+str(j)]
    ten_fold_name.append(filename)
    

###### convert image and mask to tfrecords file #######
def get_image_annotation_filename_pairs(name_list):
    name_pairs = []
    for i in name_list:
        img = image_dir + '/test' + str(int(i)) + '.png'
        mask = mask_dir + '/test' + str(int(i)) + '_Mask.png'
        name_pairs.append((img, mask))
    return name_pairs

for i in range(len(ten_fold_name)):
    image_annotation_filename_pairs = get_image_annotation_filename_pairs(ten_fold_name[i])
    write_image_annotation_pairs_to_tfrecord(filename_pairs= image_annotation_filename_pairs,
                                         tfrecords_filename='patient_cross_val-%d.tfrecords'%i)