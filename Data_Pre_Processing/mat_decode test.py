# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 18:29:22 2017

@author: Wenyuan Li
"""

import scipy.io
import numpy as np
import skimage.io as io
from PIL import Image

label_colours = [(128, 0, 0), (0, 128,0), (0, 0, 128), (128, 128, 0)]

mat_filename = './cedars-224/masks/test1_Mask.mat'

mat = scipy.io.loadmat(mat_filename, mat_dtype=True, squeeze_me=True, struct_as_record=False)

h,w = mat['ATmask'].shape

outputs = np.zeros((h,w,3), dtype = np.uint8)
num_classes = 5

img = Image.new('RGB', (h, w))
pixels = img.load()
for j_, j in enumerate(mat['ATmask']):
    for k_, k in enumerate(j):
        if k < num_classes:
            pixels[k_,j_] = label_colours[int(k-1)]
outputs = np.array(img)

infer_label = Image.fromarray(outputs)
infer_label.save()