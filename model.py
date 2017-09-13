"""
The U-Net
"""
from keras.layers import Input, RepeatVector
from keras.engine.topology import Layer
from keras.layers.core import Activation, Reshape, Lambda, Dropout
from keras.layers.convolutional import Conv3D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add, concatenate, multiply
from keras.models import Model
from keras.layers import UpSampling3D
from keras.layers.advanced_activations import PReLU
from keras import backend as K
import numpy as np


def dice_coef(y_true, y_pred):
    """
    Returns overall dice coefficient after supressing the background

    TODO : Choose channel(and axis) of background
    """
    # flatten after select only foregroudn layers (lambda)
    y_true_f = K.flatten(Lambda(lambda y_true: y_true[:, :, 1:])(y_true))
    y_pred_f = K.flatten(Lambda(lambda y_pred: y_pred[:, :, 1:])(y_pred))

    smooth = 1

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_weight(y_true, y_pred):
    """
    Returns the product of dice coefficient for each class
    it assumes channel 0 as background

    TODO : Choose channel (and axis) of background
           Choose other merge methods (sum,avg,etc)
    """

    y_true_f = (Lambda(lambda y_true: y_true[:, :, 0:])(y_true))
    y_pred_f = (Lambda(lambda y_pred: y_pred[:, :, 0:])(y_pred))

    print y_true_f.shape
    print y_pred_f.shape

    product = multiply([y_true_f, y_pred_f])

    red_y_true = K.sum(y_true_f, axis=[0, 1])
    red_y_pred = K.sum(y_pred_f, axis=[0, 1])
    red_product = K.sum(product, axis=[0, 1])

    smooth = 1
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)

    ratio_y_pred = red_y_true / (K.sum(red_y_true) + smooth)

    ratio_y_pred = 1.0 - ratio_y_pred

    ratio_y_pred = ratio_y_pred / K.sum(ratio_y_pred)

    return K.sum(multiply([dices, ratio_y_pred]))


def dice_coef_weight_r(y_true, y_pred):
    """
    Returns the product of dice coefficient for each class
    it assumes channel 0 as background

    TODO : Choose channel (and axis) of background
           Choose other merge methods (sum,avg,etc)
    """

    y_true_f = (Lambda(lambda y_true: y_true[:, :, 0:])(y_true))
    y_pred_f = (Lambda(lambda y_pred: y_pred[:, :, 0:])(y_pred))

    print y_true_f.shape
    print y_pred_f.shape

    product = multiply([y_true_f, y_pred_f])

    red_y_true = K.sum(y_true_f, axis=[0, 1])
    red_y_pred = K.sum(y_pred_f, axis=[0, 1])
    red_product = K.sum(product, axis=[0, 1])

    smooth = 1
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)

    ratio_y_pred = red_y_true / (K.sum(red_y_true) + smooth)

    ratio_y_pred = K.pow(ratio_y_pred + 0.001, -1.0)

    ratio_y_pred = ratio_y_pred / K.sum(ratio_y_pred)

    return K.sum(multiply([dices, ratio_y_pred]))


def dice_coef_weight_r_map(y_true, y_pred):
    """
    Returns the product of dice coefficient for each class
    it assumes channel 0 as background.
    This function uses an auxiliar weight_map stored in channel 0 of ground truth.
    This allows to give more importance to some regions (like borders)

    TODO : Choose channel (and axis) of background
           Choose other merge methods (sum,avg,etc)
    """

    y_true_f = (Lambda(lambda y_true: y_true[:, :, 1:])(y_true))
    y_pred_f = (Lambda(lambda y_pred: y_pred[:, :, 1:])(y_pred))
    y_map = (Lambda(lambda y_true: y_true[:, :, :1])(y_true))

    # concatenate weightmap for all the channels
    channels = y_pred.shape[2]
    y_map_c = concatenate([y_map, y_map])
    for ind in range(channels - 3):
        y_map_c = concatenate([y_map_c, y_map])

    product = multiply([y_true_f, y_pred_f])

    # apply weight map for each result
    product = multiply([product, y_map_c])
    y_true_f = multiply([y_true_f, y_map_c])
    y_pred_f = multiply([y_pred_f, y_map_c])

    red_y_true = K.sum(y_true_f, axis=[0, 1])
    red_y_pred = K.sum(y_pred_f, axis=[0, 1])
    red_product = K.sum(product, axis=[0, 1])

    smooth = 1
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)

    ratio_y_pred = red_y_true / (K.sum(red_y_true) + smooth)

    ratio_y_pred = K.pow(ratio_y_pred + 0.01, -1.0)

    ratio_y_pred = ratio_y_pred / K.sum(ratio_y_pred)

    return K.sum(multiply([dices, ratio_y_pred]))


def dice_coef_prod(y_true, y_pred):
    """
    Returns the product of dice coefficient for each class
    it assumes channel 0 as background

    TODO : Choose channel (and axis) of background
           Choose other merge methods (sum,avg,etc)
    """

    y_true_f = (Lambda(lambda y_true: y_true[:, :, 1:])(y_true))
    y_pred_f = (Lambda(lambda y_pred: y_pred[:, :, 1:])(y_pred))

    product = multiply([y_true_f, y_pred_f])

    red_y_true = K.sum(y_true_f, axis=[0, 1])
    red_y_pred = K.sum(y_pred_f, axis=[0, 1])
    red_product = K.sum(product, axis=[0, 1])

    smooth = 1
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)

    return K.prod(dices)


def dice_coef_mean(y_true, y_pred):
    """
    Returns the product of dice coefficient for each class
    it assumes channel 0 as background

    TODO : Choose channel (and axis) of background
           Choose other merge methods (sum,avg,etc)
    """

    y_true_f = (Lambda(lambda y_true: y_true[:, :, 1:])(y_true))
    y_pred_f = (Lambda(lambda y_pred: y_pred[:, :, 1:])(y_pred))

    product = multiply([y_true_f, y_pred_f])

    red_y_true = K.sum(y_true_f, axis=[0, 1])
    red_y_pred = K.sum(y_pred_f, axis=[0, 1])
    red_product = K.sum(product, axis=[0, 1])

    smooth = 1
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)

    return K.mean(dices)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def dice_coef_loss_r(y_true, y_pred):
    return (-dice_coef_weight_r_map(y_true, y_pred))


## Start defining our building blocks


def intro( nf, sz, z_sz, nch,bn):
    inputs = Input((sz, sz, z_sz, nch))
    conv = Conv3D(nf, 3, padding='same')(inputs)
    conv = PReLU(shared_axes=[1, 2, 3])(conv)
    if bn:
        conv = BatchNormalization()(conv)

    m = Model(input=inputs, output=conv)

    return m


def down_transition( nf, nconvs, sz, z_sz, nch,bn,dr):
    inputs = Input((sz, sz, z_sz, nch))

    downconv = Conv3D(nf, 2, padding='valid', strides=(2, 2, 2))(inputs)
    downconv = PReLU(shared_axes=[1, 2, 3])(downconv)
    if  bn:
        downconv = BatchNormalization()(downconv)
    if  dr:
        downconv = Dropout(0.5)(downconv)

    conv = Conv3D(nf, 3, padding='same')(downconv)
    conv = PReLU(shared_axes=[1, 2, 3])(conv)
    if  bn:
        conv = BatchNormalization()(conv)

    for _ in range( nconvs - 1):
        conv = Conv3D(nf, 3, padding='same')(conv)
        conv = PReLU(shared_axes=[1, 2, 3])(conv)
        if  bn:
            conv = BatchNormalization()(conv)

    d = add([conv, downconv])

    m = Model(input=inputs, output=d)

    return m

def up_transition(nf, nconvs, sz, z_sz, nch, nch2,bn,dr):
    input1 = Input((sz, sz, z_sz, nch))
    input2 = Input((sz * 2, sz * 2, z_sz * 2, nch2))

    upconv = UpSampling3D((2, 2, 2))(input1)
    upconv = Conv3D( nf, 2, padding='same')(upconv)
    upconv = PReLU(shared_axes=[1, 2, 3])(upconv)
    if  bn:
        upconv = BatchNormalization()(upconv)
    if  dr:
        upconv = Dropout(0.5)(upconv)

    merged = concatenate([upconv, input2])

    conv = Conv3D( nf * 2, 3, padding='same')(merged)
    conv = PReLU(shared_axes=[1, 2, 3])(conv)
    if  bn:
        conv = BatchNormalization()(conv)

    for _ in range( nconvs - 1):
        conv = Conv3D( nf * 2, 3, padding='same')(conv)
        conv = PReLU(shared_axes=[1, 2, 3])(conv)
        if  bn:
            conv = BatchNormalization()(conv)

    d = add([conv, merged])

    m = Model(input=[input1, input2], output=d)

    return m



class vnet():
    """ class that defines a volumetric u shaped volumetric network (V-Net). """

    def __init__(self, nch, sz, z_sz,nf, n_channels=3,aux_output=False,deep_supervision=0,bn=True,dr=True):

        self.sz = sz
        self.z_sz = z_sz
        self.nch = nch
        self.n_channels = n_channels
        self.ch_indx = 4
        self.aux_output = aux_output
        self.deep_supervision = deep_supervision
        self.nf = nf
        self.bn = bn
        self.dr = dr

        print 'dr:', dr
        print 'nf:', nf
        print 'bn:', bn
        print 'ds:', deep_supervision
        print 'ao;', aux_output



    # %%
    # Define the neural network
    def get_vnet(self):
       

        inputs = Input((self.sz, self.sz, self.z_sz, self.nch))

        in_tr = intro(self.nf, self.sz, self.z_sz, self.nch,self.bn)(inputs)

        #down_path

        dwn_tr1 = down_transition( self.nf * 2, 2, int(in_tr.shape[2]), int(in_tr.shape[3]), int(in_tr.shape[4]),self.bn,
                                  self.dr)(in_tr)
        dwn_tr2 = down_transition( self.nf * 4, 2, int(dwn_tr1.shape[2]), int(dwn_tr1.shape[3]), int(dwn_tr1.shape[4]), self.bn,
                                  self.dr)(dwn_tr1)
        dwn_tr3 = down_transition( self.nf * 8, 3, int(dwn_tr2.shape[2]), int(dwn_tr2.shape[3]), int(dwn_tr2.shape[4]), self.bn,
                                  self.dr)(dwn_tr2)
        dwn_tr4 = down_transition( self.nf * 16, 3, int(dwn_tr3.shape[2]), int(dwn_tr3.shape[3]), int(dwn_tr3.shape[4]), self.bn,
                                  self.dr)(dwn_tr3)


        #up_path

        up_tr4 = up_transition(self.nf * 8, 3,int(dwn_tr4.shape[2]), int(dwn_tr4.shape[3]), int(dwn_tr4.shape[4]),int(dwn_tr3.shape[4]), self.bn,self.dr)([dwn_tr4,dwn_tr3])
        up_tr3 = up_transition(self.nf * 4, 3,int(up_tr4.shape[2]), int(up_tr4.shape[3]), int(up_tr4.shape[4]),int(dwn_tr2.shape[4]), self.bn, self.dr)([up_tr4, dwn_tr2])
        up_tr2 = up_transition(self.nf * 2, 2,int(up_tr3.shape[2]), int(up_tr3.shape[3]), int(up_tr3.shape[4]),int(dwn_tr1.shape[4]), self.bn, self.dr)([up_tr3, dwn_tr1])
        up_tr1 = up_transition(self.nf * 1, 2,int(up_tr2.shape[2]), int(up_tr2.shape[3]), int(up_tr2.shape[4]),int(in_tr.shape[4]), self.bn, self.dr)([up_tr2, in_tr])





        #classification
        res = Conv3D(self.n_channels, 1, padding='same')(up_tr1)
        res = Reshape((self.sz * self.sz * self.z_sz, self.n_channels))(res)
        act = 'softmax'
        out = Activation(act, name='main')(res)

        if not self.aux_output:
            model = Model(input=inputs, output=out)

        #aux and deep supervision
        else:
            #aux_output
            aux_res = Conv3D(2, 1, padding='same')(up_tr1)
            aux_res = Reshape((self.sz * self.sz * self.z_sz, 2))(aux_res)
            aux_out = Activation(act, name='aux')(aux_res)

            outputs = [out, aux_out]

            if (self.deep_supervision > 0):
                # deep supervision#1
                deep_1 = UpSampling3D((2, 2, 2))(up_tr2)
                res = Conv3D(self.n_channels, 1, padding='same')(deep_1)
                res = Reshape((self.sz * self.sz * self.z_sz, self.n_channels))(res)

                d_out_1 = Activation(act, name='d1')(res)

                outputs.append(d_out_1)

            if (self.deep_supervision > 1):
                # deep supervision#2
                deep_2 = UpSampling3D((2, 2, 2))(up_tr3)
                deep_2 = UpSampling3D((2, 2, 2))(deep_2)
                res = Conv3D(self.n_channels, 1, padding='same')(deep_2)
                res = Reshape((self.sz * self.sz * self.z_sz, self.n_channels))(res)

                d_out_2 = Activation(act, name='d2')(res)

                outputs.append(d_out_2)

            model = Model(input=inputs, output=outputs)

        return model
