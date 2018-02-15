# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 11:24:31 2017

@author: Wenyuan Li
"""
import numpy as np
from PIL import Image


def _crop_split_image(img_org, annotation_org, height, width):
    
    imgheight, imgwidth, _ = img_org.shape
    annheight, annwidth, _ = annotation_org.shape
    assert imgwidth == annwidth and imgheight == annheight, "The image and annotation size don't match!"
    
    img = []
    ann = []
    for i in range(0, imgheight - 1, height):
        for j in range(0, imgwidth - 1, width):
            img_temp = img_org[i : i + height, j : j + height, :]
            img.append(img_temp)
            ann_temp = annotation_org[i : i + height, j : j + height, :]
            ann.append(ann_temp)
    return img, ann

img_path = './cedars-224/tiles/test1.png'
annotation_path = './cedars-224/masks_png/test1_Mask.png'

img_org = np.array(Image.open(img_path))
annotation_org = np.array(Image.open(annotation_path))

img_crop, ann_crop = _crop_split_image(img_org, annotation_org, 200, 200)

for img, annotation in zip(img_crop, ann_crop):
    height = img.shape[0]
    width = img.shape[1]
    img_vis = Image.fromarray(img)
    img_vis.show()  
    ann_vis = Image.fromarray(annotation)
    ann_vis.show()