# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:57:47 2017

@author: fferreira
"""

import numpy as np

def random_patch(scan,gt_scan = None,patch_shape=(64,128,128)):
    sh = np.array(scan.shape)
    p_sh = np.array(patch_shape)
    
    range_vals = sh - p_sh
    
    z_ = np.random.random_integers(range_vals[0])
    x_ = np.random.random_integers(range_vals[1])
    y_ = np.random.random_integers(range_vals[2])
    
    zf_ = z_+patch_shape[0]
    xf_ = x_+patch_shape[1]
    yf_ = y_+patch_shape[2]
    
    
    patch = scan[z_:zf_,x_:xf_,y_:yf_]
    
    if(gt_scan != None):
        gt_patch = gt_scan[z_:zf_,x_:xf_,y_:yf_]
        
        return patch,gt_patch
    else:
        return patch
        