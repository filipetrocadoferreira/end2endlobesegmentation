# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:20:10 2017
@author: fferreira
"""

import numpy as np
from model import vnet
from futils.util import sample_scan
from  scipy import ndimage
from futils.vpatch import deconstruct_patch,reconstruct_patch
from keras.models import model_from_json
labels = [0,4,5,6,7,8]


def one_hot_decoding(img,labels,thresh=[]):
    
    new_img = np.zeros((img.shape[0],img.shape[1]))
    r_img   = img.reshape(img.shape[0],img.shape[1],-1)

    aux = np.argmax(r_img,axis=-1)
    
    
    for i,l in enumerate(labels[1::]):
        if(thresh==[]):        
            new_img[aux==(i+1)]=l
        else:
            new_img[r_img[:,:,i+1]>thresh] = l

    return new_img



class v_segmentor(object):
    def __init__(self,batch_size=1,model='MODEL.h5',ptch_sz=128,z_sz=64,target_sz=256,target_z_sz=128):
        self.batch_size = batch_size
        self.model      = model
        self.ptch_sz    = ptch_sz   
        self.z_sz       = z_sz
        self.trgt_sz    = target_sz
        self.trgt_z_sz  = target_z_sz

        
        if(self.trgt_sz!=self.ptch_sz):
            self.patching = True
        else:
            self.patching = False
        
        model_path = model.split('.h5')[0]+'.json'
        with open(model_path, "r") as json_file:
            json_model = json_file.read()
        
            self.v = model_from_json(json_model)
       
        self.v.load_weights((self.model))
        
    def _normalize(self,scan):
        """returns normalized (0 mean 1 variance) scan"""
        
        scan = (scan - np.mean(scan))/(np.std(scan))
        return scan
    
   
    def predict(self,x):
       
        #save shape for further upload
        original_shape = x.shape        
        
        #normalize input
        x = self._normalize(x)
       
        #rescale scan to 256,256,128
        rescale_x =  sample_scan(x[:,:,:,np.newaxis],self.trgt_sz,self.trgt_z_sz)
        
        
        
        #let's patch this scan (despatchito)
        if(self.patching):
            x_patch = deconstruct_patch(rescale_x)
        else:
            x_patch = rescale_x

        del x,rescale_x      
        
       
            
        #update shape to NN - > slice axis is the last in the network
        x_patch = np.rollaxis(x_patch,1,4)
        
      
        #run predict
        pred_array = self.v.predict(x_patch,self.batch_size,verbose=0)

        # chooses our output :P (0:main pred, 1:aux output, 2-3: deep superv)
        if len(pred_array)>1:
            pred = pred_array[0]
        else:
            pred = pred_array
        
        #turn back to image shape
        pred = np.reshape(pred,(pred.shape[0],self.ptch_sz,self.ptch_sz,self.z_sz,-1))
        pred = np.rollaxis(pred,3,1)
        
        
        
        if(self.patching):
            pred = reconstruct_patch(pred)

        

        
        #one hot decoding
        masks = []
        for p in pred:
            masks.append(one_hot_decoding(p,labels))
        masks=np.array(masks,dtype='uint8')
        
        

        
        #upsample back to original shape
        zoom_seq = np.array(original_shape,dtype='float')/np.array(masks.shape,dtype='float')
        final_pred = ndimage.interpolation.zoom(masks,zoom_seq,order=0,prefilter=False)
        
       
        
        return np.reshape(final_pred,original_shape)
        
#       