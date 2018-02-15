#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:57:41 2018

@author: wenyuan
"""

import os
from xlrd import open_workbook


held_out_set = 4

excel_path = os.path.join(os.getcwd(), 'cedars-224/5_fold_partition.xlsx')

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
            
print(len(val_list), len(train_list))