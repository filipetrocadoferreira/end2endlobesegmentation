"""
The U-Net
"""
from keras.layers import Input, RepeatVector
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


class vnet():
    """ class that defines a volumetric u shaped network (V-Net). """

    def __init__(self, nch, sz, z_sz, n_channels=3):

        self.sz = sz
        self.z_sz = z_sz
        self.nch = nch
        # self.nlayers = int(np.floor(np.log(sz)/np.log(2)))+1
        self.n_channels = n_channels

        self.ch_indx = 4

    # %%
    def intro(self, nf, sz, z_sz, nch):
        '''Define intro block of vnet'''
        inputs = Input((sz, sz, z_sz, nch))
        conv = Conv3D(nf, 3, padding='same')(inputs)


        conv = PReLU(shared_axes=[1, 2, 3])(conv)
        conv = BatchNormalization()(conv)

        print 'intro',conv.shape,inputs.shape


        m = Model(input=inputs, output=conv)

        return m

    # %%

    def down_transition(self, nf, nconvs, sz, z_sz, nch):
        inputs = Input((sz, sz, z_sz, nch))
        print 'inputs',inputs
        downconv = Conv3D(nf, 2, padding='valid', strides=(2, 2, 2))(inputs)
        downconv = PReLU(shared_axes=[1, 2, 3])(downconv)
        downconv = BatchNormalization()(downconv)
        downconv = Dropout(0.5)(downconv)

        conv = Conv3D(nf, 3, padding='same')(downconv)
        conv = PReLU(shared_axes=[1, 2, 3])(conv)
        conv = BatchNormalization()(conv)

        for _ in range(nconvs - 1):
            conv = Conv3D(nf, 3, padding='same')(conv)
            conv = PReLU(shared_axes=[1, 2, 3])(conv)
            conv = BatchNormalization()(conv)


        d = add([conv, downconv])
        print d.shape, inputs.shape

        m = Model(input=inputs, output=d)

        return m

    # %%

    def up_transition(self, nf, nconvs, sz, z_sz, nch, nch2):
        input1 = Input((sz, sz, z_sz, nch))
        input2 = Input((sz * 2, sz * 2, z_sz * 2, nch2))

        upconv = UpSampling3D((2, 2, 2))(input1)
        upconv = Conv3D(nf, 2, padding='same')(upconv)
        upconv = PReLU(shared_axes=[1, 2, 3])(upconv)
        upconv = BatchNormalization()(upconv)
        upconv = Dropout(0.5)(upconv)

        merged = concatenate([upconv, input2])

        conv = Conv3D(nf * 2, 3, padding='same')(merged)
        conv = PReLU(shared_axes=[1, 2, 3])(conv)
        conv = BatchNormalization()(conv)

        for _ in range(nconvs - 1):
            conv = Conv3D(nf * 2, 3, padding='same')(conv)
            conv = PReLU(shared_axes=[1, 2, 3])(conv)
            conv = BatchNormalization()(conv)

        d = add([conv, merged])

        print 'up', d.shape, input1.shape,input2.shape,nf

        m = Model(input=[input1, input2], output=d)

        return m

    # %%
    # Define the neural network
    def get_vnet(self, nf, aux_output=False, deep_supervision=0):
        if K.image_dim_ordering() == 'th':
            inputs = Input((self.nch, self.sz, self.sz, self.z_sz))
        elif K.image_dim_ordering() == 'tf':
            inputs = Input((self.sz, self.sz, self.z_sz, self.nch))

        in_tr = self.intro(nf, self.sz, self.z_sz, self.nch)(inputs)

        dwn_tr1 = self.down_transition(nf * 2, 2, int(in_tr.shape[2]), int(in_tr.shape[3]), int(in_tr.shape[4]))(in_tr)
        dwn_tr2 = self.down_transition(nf * 4, 2, int(dwn_tr1.shape[2]), int(dwn_tr1.shape[3]), int(dwn_tr1.shape[4]))(
            dwn_tr1)
        dwn_tr3 = self.down_transition(nf * 8, 3, int(dwn_tr2.shape[2]), int(dwn_tr2.shape[3]), int(dwn_tr2.shape[4]))(
            dwn_tr2)
        dwn_tr4 = self.down_transition(nf * 16, 3, int(dwn_tr3.shape[2]), int(dwn_tr3.shape[3]), int(dwn_tr3.shape[4]))(
            dwn_tr3)
        up_tr4 = self.up_transition(nf * 8, 3, int(dwn_tr4.shape[2]), int(dwn_tr4.shape[3]), int(dwn_tr4.shape[4]),
                                    int(dwn_tr3.shape[4]))([dwn_tr4, dwn_tr3])
        up_tr3 = self.up_transition(nf * 4, 3, int(up_tr4.shape[2]), int(up_tr4.shape[3]), int(up_tr4.shape[4]),
                                    int(dwn_tr2.shape[4]))([up_tr4, dwn_tr2])
        up_tr2 = self.up_transition(nf * 2, 2, int(up_tr3.shape[2]), int(up_tr3.shape[3]), int(up_tr3.shape[4]),
                                    int(dwn_tr1.shape[4]))([up_tr3, dwn_tr1])
        up_tr1 = self.up_transition(nf, 2, int(up_tr2.shape[2]), int(up_tr2.shape[3]), int(up_tr2.shape[4]),
                                    int(in_tr.shape[4]))([up_tr2, in_tr])

        print dwn_tr4.shape

        res = Conv3D(self.n_channels, 1, padding='same')(up_tr1)

        res = Reshape((self.sz * self.sz * self.z_sz, self.n_channels))(res)

        act = 'softmax'
        out = Activation(act, name='main')(res)

        print res.shape

        if not aux_output:
            model = Model(input=inputs, output=out)


        else:
            # up_tr1_aux  = self.up_transition(nf,2,int(up_tr2.shape[2]),int(up_tr2.shape[3]),int(up_tr2.shape[4]),int(in_tr.shape[4]))([up_tr2,in_tr])
            aux_res = Conv3D(2, 1, padding='same')(up_tr1)
            aux_res = Reshape((self.sz * self.sz * self.z_sz, 2))(aux_res)

            aux_out = Activation(act, name='aux')(aux_res)

            outputs = [out, aux_out]

            if (deep_supervision > 0):
                # deep supervision#1
                deep_1 = UpSampling3D((2, 2, 2))(up_tr2)
                res = Conv3D(self.n_channels, 1, padding='same')(deep_1)
                res = Reshape((self.sz * self.sz * self.z_sz, self.n_channels))(res)

                d_out_1 = Activation(act, name='d1')(res)

                outputs.append(d_out_1)

            if (deep_supervision > 1):
                # deep supervision#2
                deep_2 = UpSampling3D((2, 2, 2))(up_tr3)
                deep_2 = UpSampling3D((2, 2, 2))(deep_2)
                res = Conv3D(self.n_channels, 1, padding='same')(deep_2)
                res = Reshape((self.sz * self.sz * self.z_sz, self.n_channels))(res)

                d_out_2 = Activation(act, name='d2')(res)

                outputs.append(d_out_2)

            model = Model(input=inputs, output=outputs)

        return model
