# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:57:47 2017

@author: fferreira
"""


import numpy as np

def random_patch(scan,gt_scan = None,patch_shape=(64,128,128)):
    sh = np.array(scan.shape)
    p_sh = np.array(patch_shape)
    
    
    
    range_vals = sh[0:3] - p_sh 
    
   
    
    origin = [np.random.random_integers(x) for x in range_vals]
    finish  = origin + p_sh
    
    
    
    
    idx = [np.arange(o_,f_) for o_,f_ in zip(origin,finish)]
    
       
    patch = scan[np.ix_(idx[0],idx[1],idx[2])]

    
    if(gt_scan != None):
        gt_patch = gt_scan[np.ix_(idx[0],idx[1],idx[2])]
        
        return patch,gt_patch
    else:
        return patch
        
def deconstruct_patch(scan,patch_shape=(64,128,128),stride = 0.25):
    
    sh = np.array(scan.shape,dtype=int)
    p_sh = np.array(patch_shape,dtype=int)
    
    if stride == -1:
        stride = p_sh
    elif isinstance(stride, float):
        stride  = p_sh * stride
    else:
        stride  = np.ones(3)*stride
    
    stride = stride.astype(int)
    
    
    
    n_patches =   (sh[0:3] - p_sh + stride)  / stride 
    

    
    patches = []
    
    for z,x,y in np.ndindex(tuple(n_patches)):
        it = np.array([z,x,y],dtype= int)
        origin = it * stride
        finish = it * stride + p_sh        
        
        idx = [np.arange(o_,f_) for o_,f_ in zip(origin,finish)]
        
       
        patch = scan[np.ix_(idx[0],idx[1],idx[2])]
        
        patches.append(patch)
    
    patches = np.array(patches)    
    return patches
    
def reconstruct_patch(scan,original_shape=(128,256,256),stride = 0.25):
    
    p_sh = np.array(scan.shape,dtype=int)[1:4]
    sh = np.array(original_shape,dtype=int)
    
    if stride == -1:
        stride = p_sh
    elif isinstance(stride, float):
        stride  = p_sh * stride
    else:
        stride  = np.ones(3)*stride
    
    stride = stride.astype(int)
       
    
    n_patches =   (sh - p_sh + stride)  / stride 
    
 
    
    
    #init result
    result = np.zeros(tuple(sh)+(scan.shape[-1],),dtype=float)
    
   
    
    index = 0
    for z,x,y in np.ndindex(tuple(n_patches)):
        it = np.array([z,x,y],dtype= int)
        origin = it * stride
        finish = it * stride + p_sh        
        
        idx = [np.arange(o_,f_) for o_,f_ in zip(origin,finish)]
        
        
       
        result[np.ix_(idx[0],idx[1],idx[2])] += scan[index]
        
       
        
        index +=1
      
    #normalize (0-1) output
    r_sum = np.sum(result,axis=-1)

    r_sum[r_sum==0] = 1 #hmm this should not be necessary - but it is :(
    r_sum = np.repeat(r_sum[:,:,:,np.newaxis],scan.shape[-1],-1) #repeat the sum along channel axis (to allow division)
    
    result = np.divide(result,r_sum) 
    
    return result
    