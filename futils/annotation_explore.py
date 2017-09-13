# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:56:22 2017

@author: fferreira
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:37:39 2017

@author: Marlene Machado


"""


import numpy as np
from skimage import draw




def get_coordinates(lines,UID):
    return lines[lines[:,0]==UID][:,1::].astype(float)
    
#%%

def world_to_vx(point,origin,spacing):
    return np.multiply((point - origin),1/spacing)
#%%
def paint_node(vx_point,vx_dims,mask):
    

    for i in range(int(-vx_dims[0]/2),int(vx_dims[0]/2)+1):
        
        r = vx_dims[1]/2-abs(i)
        rr, cc = draw.circle(vx_point[1],vx_point[2],r)
        mask[int(vx_point[0]+i),rr,cc] = 1
       
    
    return mask
        
    

#%%    
def get_mask(seg_scan,UID,spacing,origin,file='annotations.csv'):
    mask=np.zeros(seg_scan.shape)
    
    
    with open(file,'rb') as f:
        
        lines = np.genfromtxt(f, delimiter=',',dtype=str)[1::]
        coordinates = get_coordinates(lines,UID)
        for coord in coordinates:
            point = coord[0:3]
            diam  = coord[3]
            
            
            #%change coordinate system (x,y,z) - > (z,y,x)
            #point = np.roll(point,1)
            point = point[::-1]
            
            #%convert to voxel coordinate system                   
            vx_point = world_to_vx(point,origin,spacing)
            #%and get the dimensions in scan space
            vx_dim = np.multiply(1/spacing,diam)
            
            print(vx_point,vx_dim[0])    
            
            mask = paint_node(vx_point,vx_dim,mask)
        

    return mask
