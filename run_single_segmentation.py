import time
import futils.util as futil
import segmentor as v_seg
import keras.backend as K

K.set_learning_phase(1)

#LOAD THE MODEL
segment = v_seg.v_segmentor(batch_size=1, model='models/final.h5', ptch_sz=128, z_sz=64)



#LOAD THE CT_SCAN
scan_file = 'VESSEL12_15.mhd'
ct_scan, origin, spacing, orientation = futil.load_itk('data/val/A/'+scan_file, get_orientation=True)
if (orientation[-1] == -1):
    ct_scan = ct_scan[::-1]
print 'Origem: '
print origin
print 'Spacing: '
print spacing
print 'Orientation: '
print orientation



#NORMALIZATION
ct_scan = futil.normalize(ct_scan)


#PREDICT the segmentation
t1 = time.time()
lobe_mask = segment.predict(ct_scan)
t2 = time.time()
print t2-t1


#Save the segmentation
futil.save_itk('results/'+scan_file, lobe_mask, origin, spacing)
