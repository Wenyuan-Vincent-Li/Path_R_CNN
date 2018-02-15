#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:40:05 2018

@author: wenyuan
"""
import numpy as np
import scipy.io
import os
import sys

def find_union_set(mask, bar = None):
    r, c = mask.shape
    my_union = Unionfind(r * c, mask.reshape(-1))
    
    dr = [0, 1, -1, 0] # right, down, up, left
    dc = [1, 0, 0, -1]
    old = np.ones(r * c)

    while(not np.array_equal(my_union.father,old)):   
        old = np.copy(my_union.father)
        for i in range(r):
            for j in range(c):
                if(mask[i, j] == 4):
                    my_union.set_father(i * c + j, -1)
                else:
                    for k in range(4):
                        if (i + dr[k] >= 0 and i + dr[k] < r and
                           j + dc[k] >= 0 and j + dc[k] < c and 
                           mask[i + dr[k], j + dc[k]] == mask[i, j]):
                            my_union.connect(i * c + j, 
                                             (i + dr[k]) * c + j + dc[k])
    return my_union.father.reshape((r, c))

def generate_instance_mask(grouped_mask, orig_mask, mode = -1):
    """ stack the grouped_mask to image_height * image_width * instance_num
    return the instance mask and class_ids
    """
    unique_set = np.copy(np.unique(grouped_mask))
    class_ids = []
    if mode == -1:
        hash_dict = {'4': 0, '1': 1, '2': 2, '3': 3} # for sementic 144 mode skip this lable change
    else:
        hash_dict = {'0': 0, '1': 1, '2': 2, '3': 3}
    
    for index, val in enumerate(unique_set):
        if index == 0:
            stack = (grouped_mask == val).astype(int)
        else:
            stack = np.dstack((stack, (grouped_mask == val).astype(int)))
        pos = np.argwhere(grouped_mask == val)[0]
        class_ids.append(hash_dict[str(orig_mask[pos[0], pos[1]])])
    return stack, class_ids

def save_instance_mask(stack, class_ids, image_id, dir_name = 'cedars-224/masks_instance/',\
                      mode = -1, patch = -1):
    """ save the instance mask
    """
    image_id = str(image_id)
    if mode == -1:
        mat_filename = image_id.zfill(4) + '_instance'
        save_path = os.path.join(os.getcwd(), dir_name + mat_filename)
    else:
        mat_filename = image_id.zfill(4) + '_' + str(patch).zfill(4) + '_instance'
        dir_name = 'cedars-224/masks_instance_mod_' + str(mode) + '/'
    
    save_path = os.path.join(os.getcwd(), dir_name + mat_filename)
    res_dict = {'segmentation': stack, 'class_ids': class_ids}
    scipy.io.savemat(save_path, res_dict)
    
def instance_mask_generator(org_mask, image_id, mode = -1, patch = -1):
    """Generating the instance mask from the sementic mask
    """
    grouped_mask = find_union_set(org_mask)
    instance_mask, class_ids = generate_instance_mask(grouped_mask, org_mask, mode = mode)
    save_instance_mask(instance_mask, class_ids, image_id, mode = mode, patch = patch)
    
def modified_instance_mask(mask, class_ids, image_id, th = 999, 
                           dir_name = 'cedars-224/masks_instance_mod/'):
    # delete the instance with area smaller than th
    try:
        _,_, d = mask.shape
    except ValueError:
        d = 1
        class_ids = [class_ids]
        mask = np.expand_dims(mask, axis=-1)
    
    delete_list = []
    for i in range(d):
        if sum(sum(mask[:,:,i])) < th:
            delete_list.append(i)    
    mask = np.delete(mask, delete_list, axis = 2)
    class_ids = np.delete(class_ids, delete_list).tolist()
    save_instance_mask(mask, class_ids, image_id, dir_name)

class ShowProcess():
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 0 # 当前的处理进度
    max_steps = 0 # 总共需要处理的次数
    max_arrow = 50 #进度条的长度

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #计算显示多少个'>'
        num_line = self.max_arrow - num_arrow #计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps #计算完成进度，格式为xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r' #带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar) #这两句打印字符到终端
        sys.stdout.flush()

    def close(self, words='done'):
        print('')
        print(words)
        self.i = 0
 


class Unionfind:
    
    def __init__(self, n, mask):
        self.father = np.arange(n)
    
    def _find(self, x):
        if self.father[x] == x: 
            return x
        self.father[x] = self._find(self.father[x])
        return self.father[x]
    
    def connect(self, a, b):
        root_a = self._find(a)
        root_b = self._find(b)
        if (root_a != root_b):
            self.father[root_b] = root_a
            
    def set_father(self, pos, val):
        self.father[pos] = val