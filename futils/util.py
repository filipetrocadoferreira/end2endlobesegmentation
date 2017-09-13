# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 09:35:14 2017

@author: fferreira
"""
import os
import pickle
import numpy as np
import collections
from skimage import color,transform
import SimpleITK as sitk
from scipy import ndimage
from skimage.io import imsave
import nrrd


MIN_BOUND = -1000.0
MAX_BOUND = 400.0
#%%
def get_UID(file_name):
    print(file_name)
    if(os.path.isfile(file_name)):
        file = open(file_name,'rb')    
        try:        
            data = pickle.load(file)
            print(data)
            file.close()
            
            return data
        except Exception as inst:
            print type(inst)     # the exception instance
      
            print inst           # __str__ allows args to be printed directly

            print('no pickle here')
            return [],[],[]
    print 'nop'
    return[],[],[]
   
#%%    
def get_scan(file_name):
    if(os.path.isfile(file_name)):
        file = open(file_name,'rb')    
        data = pickle.load(file)
        file.close()
        return np.rollaxis(data,2)[::-1]
    else:
        return[],[],[]
#%%
def load_itk(filename,original=False,get_orientation=False):
    # Reads the image using SimpleITK
    if(os.path.isfile(filename) ):
        itkimage = sitk.ReadImage(filename)
        
    else:
        print 'nonfound:',filename
        return [],[],[]

    # Convert the image to a  numpy array first ands then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
        
    #ct_scan[ct_scan>4] = 0 #filter trachea (label 5)
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    if(get_orientation):
        orientation = itkimage.GetDirection()
        return ct_scan, origin, spacing,orientation

    
    
    return ct_scan, origin, spacing
#%%
def save_itk(filename,scan,origin,spacing,dtype = 'uint8'):
    
        
        stk = sitk.GetImageFromArray(scan.astype(dtype))
        stk.SetOrigin(origin[::-1])
        stk.SetSpacing(spacing[::-1])
        
        
        writer = sitk.ImageFileWriter()
        writer.Execute(stk,filename,True)
#%%

def load_nrrd(filename):
        readdata, options = nrrd.read(filename)
        origin = np.array(options['space origin']).astype(float)
        spacing = np.array(options['space directions']).astype(float)
        spacing = np.sum(spacing,axis=0)
        return np.transpose(np.array(readdata).astype(float)),origin[::-1],spacing[::-1]
                
#%% Save in _nii.gz format
def save_nii(dirname,savefilename,lung_mask):    
    
    array_img = nib.Nifti1Image(lung_mask, affine=None, header=None)
    nib.save(array_img, os.path.join(dirname,savefilename))
    
def save_slice_img(folder,scan,uid):
    print(uid,scan.shape[0])    
    for i,s in enumerate(scan):
        imsave(os.path.join(folder,uid+'sl_'+str(i)+'.png'),s)
#%%
def normalize(image,min_=MIN_BOUND,max_=MAX_BOUND):
    image = (image - min_) / (max_ - min_)
    image[image>1] = 1.
    image[image<0] = 0.
    return image
#%%
def dice(seg,gt):
    
    im1 = np.asarray(seg).astype(float)
    im2 = np.asarray(gt).astype(float)

    
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    
    intersection = (im1*im2).sum()
   
    return 2. * intersection / (im1.sum() + im2.sum())    
#%%
def dice_mc(seg,gt,labels=[]):
    if(labels==[]):   
        labels= np.unique(gt)
    
    dices = np.zeros(len(labels))    
    for i,l in enumerate(labels):
        im1 = seg==l
        im2 = gt ==l
        
        dices[i] = dice(im1,im2)
    
    return dices    
    
   

#%%
def recall(seg,gt):
    
    im1 = np.asarray(seg>0).astype(np.bool)
    im2 = np.asarray(gt>0).astype(np.bool)

    
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2).astype(float)

    
    if(im2.sum()>0):
        return intersection.sum() / ( im2.sum())    
    else:
        return 1.0    
        
        
def sample_scan(scan,sz = 128,z_length=64,pivot_axis=1,scale_axis=0,order=0):
        """Downscale scan only in slicing direction with nearest interpolation """
        zoom_seq = np.array([z_length,sz,sz,1],dtype='float')/np.array(scan.shape,dtype='float')
        s = ndimage.interpolation.zoom(scan,zoom_seq,order=order,prefilter=order)
        
        return s

def _one_hot_enc(patch,input_is_grayscale = False,labels=[]):

        labels = np.array(labels)
        N_classes = labels.size
       
             
        
        ptch_ohe = np.zeros((patch.shape[0],patch.shape[1])+(N_classes,))
        for i,l in enumerate(labels):
            
           m = np.where((patch == l).all(axis=2))
            
           new_val = np.zeros(N_classes)
           new_val[i] = 1.
            
            

           ptch_ohe[m] = new_val           
        
        return ptch_ohe
        
def weight_map(label,labels =[0,4,5,6,7,8]):
    
    
    #one hot encoding of label map
    gt_cat = []
    for gt in label:
        gt_cat.append(_one_hot_enc(gt[:,:,np.newaxis],False,labels))    
    gt_cat = np.array(gt_cat)
   
    
    #fill holes and erode to have borders:
    for i in range(1,gt_cat.shape[-1]):
        #gt_cat[:,:,:,i] = ndimage.morphology.binary_fill_holes(gt_cat[:,:,:,i])    
        gt_cat[:,:,:,i] = ndimage.binary_erosion(gt_cat[:,:,:,i],iterations=5)
    
    #create image back from hot encoding
    new_gt= np.zeros_like(label)
    for i,l in enumerate(labels[1::]):
        new_gt[gt_cat[:,:,:,i+1]==1]=l
   
    
    #create weight map
    borders = np.zeros_like(new_gt)    
    borders[(label>0)&(new_gt<1)] = 1.0
    
    weightmap = borders
    #gaussian filter to smooth
    weightmap = ndimage.filters.gaussian_filter(borders.astype('float'),3)
    
   
    
    return weightmap
#%%
def get_fissures(scan):
    lung = np.zeros_like(scan)
    lung[scan>0] = 1
    
    weightmap = weight_map(scan)
    
    weightmap2 = weight_map(lung,labels=[0,1])
    
    weightmap = weightmap-weightmap2
    
    weightmap[weightmap<0] = 0
    
    return weightmap