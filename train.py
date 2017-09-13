from keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from util.patch import Patches
from dataloader import TwoScanIterator
from model import vnet, dice_coef, dice_coef_loss, dice_coef_loss_r, dice_coef_mean, dice_coef

from keras.optimizers import Adam, SGD
from keras import callbacks
from keras.utils import plot_model
import time
from futils.util import dice
import argparse
import os

### set important paths

dir_path = os.path.dirname(os.path.realpath(__file__))

path_model = os.path.join(dir_path,'models')
test_dir =  os.path.join(dir_path,'data/train')
val_dir = os.path.join(dir_path,'data/val')

K.set_learning_phase(1)  # try with 1

parser = argparse.ArgumentParser(
    description='End2End Supervised Lobe Segmentation')

parser.add_argument(
    '-path',
    '--path',
    help='Model Path',
    type=str,
    default='/models/')
parser.add_argument(
    '-train',
    '--train_dir',
    help='Train Data',
    type=str,
    default='/data/train')

parser.add_argument(
    '-val',
    '--val_dir',
    help='Validation Data',
    type=str,
    default='/data/val')

parser.add_argument(
    '-lr',
    '--lr',
    help='learning rate',
    type=float,
    default=0.001)

parser.add_argument(
    '-load',
    '--load',
    help='load last model',
    type=int,
    default=0)

parser.add_argument(
    '-aux',
    '--aux_output',
    help='Value of Auxiliary Output',
    type=float,
    default=0.5)

parser.add_argument(
    '-ds',
    '--deep_supervision',
    help='Number of Deep Supervisers',
    type=int,
    default=2)

parser.add_argument(
    '-fs',
    '--feature_size',
    help='Number of initial of conv channels',
    type=int,
    default=16)

parser.add_argument(
    '-bn',
    '--batch_norm',
    help='Set Batch Normalization',
    type=int,
    default=1)

parser.add_argument(
    '-dr',
    '--dropout',
    help='Set Dropout',
    type=int,
    default=1)



def train(args):


    # our experiment name
    str_name = str(time.time()) + '_' + str(args.lr) + str(args.load)  + 'a_o_' + str(
        args.aux_output) +'ds' + str(args.deep_supervision) + 'dr' + str(args.dropout)+'bn'+str(args.batch_norm) +'fs'+str(args.feature_size)


    # trgt is the target dimension of the slice of the resized scan during dataloading
    trgt = 256
    # z_trgt = number of slices of a scan during dataloading
    z_trgt = 128

    # ptch_sz is the dimension of the patch dimension used during train
    ptch_sz = 128
    # length is the number of slices used during training
    ptch_z_sz = 64

    # stop is the number of epochs needed to stop in case no improvment (when negative is not applied)
    stop = -1

    # create model with auxiliary output (we want always this)
    bool_aux_output = True
    # ratio of aux_output in the loss
    aux_output = args.aux_output

    # nr of deep supervision outputs
    deep_supervision = args.deep_supervision

    # training vars:
    batch_size = 1
    nb_epoch = 4000
    labels = [0, 4, 5, 6, 7, 8]  # our output classes
    input_channels = 1
    epoch_iterations = 20

    learning_rate = args.lr

    if trgt == ptch_sz:
        patching = False
    else:
        patching = True




    #initial loss
    w_ = [1]
    loss_ = [dice_coef_loss_r]
    metrics_ = [dice_coef, dice_coef_mean]

    #add auxiliary output (we create always the structure and then give weight zero to disable) -> not good for performance
    if bool_aux_output:
        loss_.append(dice_coef_loss)
        w_ = [(1 - aux_output), (aux_output)]

    #setting deep_supervision losses&weights
    if (deep_supervision > 2):
        deep_supervision = 2
    if (abs(deep_supervision) > 0):
        if (deep_supervision > 0):
            ratio = ((deep_supervision + 1) / (2.0 + deep_supervision))

            w_ = np.asarray(w_) * ratio
            w_ = w_.tolist()
            print w_
            for _ in range(abs(deep_supervision)):
                w_.append((1 - ratio) / deep_supervision)
                loss_.append(dice_coef_loss_r)
        else:
            for _ in range(abs(deep_supervision)):
                w_.append(0)
                loss_.append(dice_coef_loss_r)



    # our optimization
    optim = Adam(lr=learning_rate)

    # Define the Model
    vn = vnet(input_channels, ptch_sz, ptch_z_sz, args.feature_size, n_channels=len(labels), aux_output=bool_aux_output,
              deep_supervision=abs(deep_supervision),bn=args.batch_norm,dr=args.dropout)
    net = vn.get_vnet()  # u net using upsample
    net.compile(optimizer=optim, loss=loss_, metrics=metrics_, loss_weights=w_)

    #load last model
    if (args.load > 0):
        net.load_weights((path_model+'/3dlobesweights.best.hdf5'))


    # save model config
    model_json = net.to_json()
    with open((path_model+'/' + str_name + 'lobesMODEL.json'), "w") as json_file:
        json_file.write(model_json)

    ###our data iterators
    train_it = TwoScanIterator(test_dir, batch_size=batch_size, c_dir_name='D',
                               fnames_are_same=True, target_size=(trgt, trgt),
                               shuffle=True, is_a_grayscale=True, is_b_grayscale=False, is_b_categorical=True,
                               rotation_range=0.05, height_shift_range=0.05, slice_length=z_trgt,
                               width_shift_range=0.05, zoom_range=0.05,
                               horizontal_flip=False, vertical_flip=False,
                               fill_mode='constant', cval=-1, separate_output=bool_aux_output,
                               deep_supervision=abs(deep_supervision),
                               patch_divide=patching, ptch_sz=ptch_sz, patch_z_sz=ptch_z_sz, ptch_str=-1,
                               labels=labels)

    val_it = TwoScanIterator(val_dir, batch_size=batch_size, c_dir_name='D',
                             fnames_are_same=True, target_size=(trgt, trgt),
                             shuffle=True, is_a_grayscale=True, is_b_grayscale=False, is_b_categorical=True,
                             rotation_range=0.00, height_shift_range=0.00, slice_length=z_trgt,
                             width_shift_range=0.00, zoom_range=0.0,
                             horizontal_flip=False, vertical_flip=False, separate_output=bool_aux_output,
                             deep_supervision=abs(deep_supervision),
                             fill_mode='constant', cval=-1,
                             patch_divide=patching, ptch_sz=ptch_sz, patch_z_sz=ptch_z_sz, ptch_str=-1, labels=labels)

    #early stop?
    if (stop < 0):
        stop = nb_epoch #if not, we set to the length of training

    # callbacks
    checker = callbacks.ModelCheckpoint(path_model+'/3dlobesweights.best.hdf5', monitor='loss',
                                        verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    saver = callbacks.ModelCheckpoint(path_model+'/' + str_name + 'lobesMODEL.h5', monitor='loss', verbose=1,
                                      save_best_only=True, save_weights_only=True, mode='auto', period=1)
    tb = callbacks.TensorBoard(log_dir=dir_path+'/logs/' + str_name, histogram_freq=10,
                               write_graph=False, write_images=True)
    stopper = callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=stop, verbose=0, mode='auto')

    #training our network :)
    net.fit_generator(train_it.generator(), epoch_iterations * batch_size, nb_epoch=nb_epoch, verbose=1,
                      validation_data=val_it.generator(), nb_val_samples=3,
                      callbacks=[checker, tb, stopper, saver])

    print 'finish train: ', str_name


if __name__ == '__main__':
    train(parser.parse_args())
