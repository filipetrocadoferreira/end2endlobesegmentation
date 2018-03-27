# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:20:10 2017

@author: fferreira
"""

import numpy as np
from util.vnet_triclass import vnet
from futils.util import sample_scan
from  scipy import ndimage
from skimage import morphology
from futils.vpatch import deconstruct_patch,reconstruct_patch
from keras.models import model_from_json
labels = [0,4,5,6,7,8]


def categorical_to_gray(img,labels,thresh=[]):
    
    new_img = np.zeros((img.shape[0],img.shape[1]))
    r_img   = img.reshape(img.shape[0],img.shape[1],-1)

    aux = np.argmax(r_img,axis=-1)
    
    
    for i,l in enumerate(labels[1::]):
        if(thresh==[]):        
            new_img[aux==(i+1)]=l
        else:
            new_img[r_img[:,:,i+1]>thresh] = l
        
    #new_img[0,0] = labels[-1]
    #print(np.unique(new_img))
    return new_img

def filter_scan(scan):
    
   
    lung = scan[:,:,:,0]<0.2
    
    c_l= ndimage.measurements.center_of_mass(lung)
    
    
    print c_l
    
    

    for i in range(1,scan.shape[-1]):
    
        channel = scan[:,:,:,i]
        #channel_ = ndimage.binary_erosion(channel>0.08,iterations=2)
        
        
        
        #channel[lung==0]=0

    
        labeled_array, num_features = ndimage.label(channel>0.3)
        
        
    
        if(num_features>1):
            sizes = ndimage.sum(channel,labeled_array,index=range(0,num_features))        
            ind = np.argmax(sizes) 
            
            size_ratio = sizes/np.sum(sizes)
            
            print size_ratio[ind]
                     
            if(size_ratio[ind]>0.5):
                scan[:,:,:,i] *= labeled_array==(ind)
            else:
                new_inds = np.array(range(0,num_features))[size_ratio>0.15]
                print 'new',size_ratio[new_inds]
                centers = ndimage.measurements.center_of_mass(channel,labeled_array,tuple(new_inds))
                centers = np.array(centers)-np.array(c_l)
                norms = [np.linalg.norm(x) for x in centers]
                
                #select object closer to lung centroid
                
                min_dist = new_inds[np.argmin(norms)]
                if min_dist:
                    scan[:,:,:,i] *= labeled_array==(min_dist)
                else:
                    scan[:,:,:,i] = np.zeros_like(scan[:,:,:,i])
                
           
            
    
    #normalize probabilities
    lung_scan = scan[:,:,:,1::]
#    
#    
#    sum_scan = (1-scan[:,:,:,0])
#    sum_scan = np.repeat(sum_scan[:,:,:,np.newaxis],lung_scan.shape[-1],-1) #repeat the sum along channel axis (to allow division)
#   
#    lung_scan/=sum_scan
##    
#    lung_scan = scan[:,:,:,0::]
    
    sum_scan = np.sum(lung_scan,axis=-1)
    sum_scan[sum_scan==0] = 1
    sum_scan = (1-scan[:,:,:,0])/sum_scan
    sum_scan = np.repeat(sum_scan[:,:,:,np.newaxis],lung_scan.shape[-1],-1) #repeat the sum along channel axis (to allow division)
       
   
    lung_scan*=sum_scan
    
    
    return scan

def filter_lung_borders(lung_borders,return_full_lung = True):
    
    labeled_array, num_features = ndimage.label(lung_borders>0)
    
    if(num_features>1):
        sizes = ndimage.sum(lung_borders>0,labeled_array,index=range(1,num_features+1))        
        sizes,indexes = zip(*sorted(zip(sizes,range(1,num_features+1)),reverse=True))
        sizes = sizes/np.sum(sizes)
        
        lung_indexes = []
        if(sizes[0]>0.8):
            
            lung_indexes = indexes[0:1]
            
        
        elif ((sizes[0]+sizes[1])>0.7):
            
            lung_indexes = indexes[0:2]
            
        else:
            return lung_borders
        
       
        
        mask =  np.in1d(labeled_array.flatten(), lung_indexes).reshape(lung_borders.shape)
        
        lung_borders = np.zeros_like(lung_borders)
        lung_borders[mask] = 1
                
        lung_borders_d = ndimage.morphology.binary_dilation(lung_borders>0,iterations=7)
        if(return_full_lung):
            
            full_lung = ndimage.binary_fill_holes(lung_borders_d).astype(int)
            print np.sum(full_lung),np.sum(lung_borders)
            return full_lung,lung_borders            
            
    
    return lung_borders
            
def filter_lobes_with_borders(lobes,lung_borders,full_lung):
    markers = np.zeros_like(lobes)
    labels = np.unique(lobes)
     #erode full lung mask
    full_lung_ = ndimage.morphology.binary_erosion(full_lung>0,iterations=19)
        
    
    for l in labels[1::]:
        print l 
        lobe = lobes==l
        
        lobe = ndimage.morphology.binary_erosion(lobe,iterations=1)
        lobe[full_lung_==0] = 0
        labeled_array, num_features = ndimage.label(lobe>0.0)
        
        if(num_features>1):
            sizes = ndimage.sum(lobe,labeled_array,index=range(0,num_features+1))        
            ind = np.argmax(sizes)
            lobe[labeled_array!=ind] = 0
        
        markers +=  lobe * l
    
    #create distance transform from borders    
    borders_d = ndimage.morphology.binary_dilation(lung_borders>0,iterations=1)
    d_borders = ndimage.morphology.distance_transform_edt(borders_d)
    
    weightmap = d_borders.astype(int)
    
    full_lung = ndimage.morphology.binary_erosion(full_lung>0,iterations=8)
    
    lobes = morphology.watershed(weightmap.astype('uint8'), markers.astype(int),mask=(full_lung>0))
    return lobes

class v_segmentor(object):
    def __init__(self,batch_size=1,model='MODEL.h5',ptch_sz=128,z_sz=64,target_sz=256,target_z_sz=128,filtering=False,process_fissure=True):
        self.batch_size = batch_size
        self.model      = model
        self.ptch_sz    = ptch_sz   
        self.z_sz       = z_sz
        self.trgt_sz    = target_sz
        self.trgt_z_sz  = target_z_sz
        self.filtering  = filtering
        self.process_fissure= process_fissure
        
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
        
        
        if (self.process_fissure) & (len(pred_array)>1):
            pred = pred_array[1]
        
            #turn back to image shape
            pred = np.reshape(pred,(pred.shape[0],self.ptch_sz,self.ptch_sz,self.z_sz,-1))
            pred = np.rollaxis(pred,3,1)
            
            
            
            if(self.patching):
                pred = reconstruct_patch(pred)
            
            
            
            #convert from categorical to uint8
            masks = []
            for p in pred:
                masks.append(categorical_to_gray(p,[0,1]))
            
            masks=np.array(masks,dtype='uint8')
            
            
            
            lung_borders = masks
            
            #convert from categorical to uint8
            masks = []
            for p in pred:
                masks.append(categorical_to_gray(p,[0,1],thresh=0.35))
            
            masks=np.array(masks,dtype='uint8')
            
            lung_borders_aux = masks
            
            full_lung,_ = filter_lung_borders(lung_borders_aux)
            
            lung_borders[full_lung==0] = 0
            
            #upsample back to original shape
            zoom_seq = np.array(original_shape,dtype='float')/np.array(lung_borders.shape,dtype='float')
            final_pred_fissure = ndimage.interpolation.zoom(lung_borders,zoom_seq,order=0,prefilter=False)
        
        if len(pred_array)>1:
            pred = pred_array[0]
        else:
            pred = pred_array
        
        #turn back to image shape
        pred = np.reshape(pred,(pred.shape[0],self.ptch_sz,self.ptch_sz,self.z_sz,-1))
        pred = np.rollaxis(pred,3,1)
        
        
        
        if(self.patching):
            pred = reconstruct_patch(pred)
            
        print pred.shape
        
        if(self.filtering):
            
            #pred[full_lung==0] = [1,0,0,0,0,0]
            pred = filter_scan(pred)
        
        #convert from categorical to uint8
        masks = []
        for p in pred:
            masks.append(categorical_to_gray(p,labels))
        
        masks=np.array(masks,dtype='uint8')
        
        
        #masks = filter_lobes_with_borders(masks,lung_borders,full_lung)
        
        #upsample back to original shape
        zoom_seq = np.array(original_shape,dtype='float')/np.array(masks.shape,dtype='float')
        final_pred = ndimage.interpolation.zoom(masks,zoom_seq,order=0,prefilter=False)
        
       
    
        return np.reshape(final_pred,original_shape)
        
#       