"""Auxiliar methods to deal with loading the dataset."""
import os
import random
import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator, load_img, img_to_array
from keras.preprocessing.image import apply_transform, flip_axis
from keras.preprocessing.image import transform_matrix_offset_center
from skimage import color, transform
from futils.vpatch import random_patch
import futils.util as futil
from  scipy import ndimage
import time
import glob

"""3D Extension of the work in https://github.com/costapt/vess2ret/blob/master/util/data.py"""

class TwoScanIterator(Iterator):
    """Class to iterate A and B 3D scans (mhd or nrrd) at the same time."""

    def __init__(self, directory, a_dir_name='A', b_dir_name='B', c_dir_name=None,
                 fnames_are_same=True,
                 is_a_binary=False, is_b_binary=False, is_a_grayscale=False,
                 is_b_grayscale=False, a_extension='.mhd', b_extension='.nrrd', c_extension='.mhd',
                 is_b_categorical=False, target_size=(512, 512),
                 rotation_range=0., height_shift_range=0., width_shift_range=0.,
                 shear_range=0., zoom_range=0., fill_mode='constant', cval=0.,
                 horizontal_flip=False, vertical_flip=False, sequence_flip=False, slice_sample=0.5, slice_length=-1,
                 dim_ordering=K.image_dim_ordering(),
                 N=-1, batch_size=1, shuffle=True, seed=None, weight_map = 10,
                 patch_divide=False, ptch_sz=64, patch_z_sz=16, ptch_str=32, patches_per_scan=5, separate_output=False,
                 deep_supervision=0, labels=[]):
        """
        Iterate through two directories at the same time.

        Files under the directory A and B with the same name will be returned
        at the same time.
        Parameters:
        - directory: base directory of the dataset. Should contain two
        directories with name a_dir_name and b_dir_name;
        - a_dir_name: name of directory under directory that contains the A
        images;
        - b_dir_name: name of directory under directory that contains the B
        images;
        - c_dir_name : this is the auxiliar output folder
        - a/b/c_extension : type of the scan: nrrd or mhd (no dicom available)

        - is_a_binary: converts A images to binary images. Applies a threshold of 0.5.
        - is_b_binary: converts B images to binary images. Applies a threshold of 0.5.
        - is_a_grayscale: if True, A images will only have one channel.
        - is_b_grayscale: if True, B images will only have one channel.
        - N: if -1 uses the entire dataset. Otherwise only uses a subset;
        - batch_size: the size of the batches to create;
        - shuffle: if True the order of the images in X will be shuffled;
        - seed: seed for a random number generator;
        - slice_length: nr of slices to include in the scan (-1 to use sample)
        - slice_sample: sample the correspondent ratio of slices in the scan
        - separate_output : to use when using auxiliar output
        - deep_supervision: number of deep classifiers
        - labels: output classes
        - weight_map = value to give to lung borders

        """
        self.directory = directory

        self.a_dir = os.path.join(directory, a_dir_name)
        self.b_dir = os.path.join(directory, b_dir_name)

        self.c_dir = None
        self.aux = False
        if c_dir_name is not None:
            self.c_dir = os.path.join(directory, c_dir_name)
            self.aux = True

        self.a_extension = a_extension
        self.b_extension = b_extension
        self.c_extension = c_extension

        a_files = set(x.split(a_extension)[0].split(self.a_dir + '/')[-1] for x in
                      glob.glob(self.a_dir + '/*' + self.a_extension))
        b_files = set(x.split(b_extension)[0].split(self.b_dir + '/')[-1] for x in
                      glob.glob(self.b_dir + '/*' + self.b_extension))


        ##print(a_files)
        ##print(b_files)

        if fnames_are_same is True:
            # Files inside a and b should have the same name. Images without a pair
            # are discarded.
            self.filenames = list(a_files.intersection(b_files))
            self.b_fnames = self.filenames
            if self.c_dir is not None:
                self.c_fnames = self.filenames

        else:
            self.filenames = sorted(os.listdir(self.a_dir))
            self.a_fnames = sorted(os.listdir(self.a_dir))
            self.b_fnames = sorted(os.listdir(self.b_dir))
            if self.c_dir is not None:
                self.c_fnames = sorted(os.listdir(self.b_dir))
        # Use only a subset of the files. Good to easily overfit the model
        if N > 0:
            random.shuffle(self.filenames)
            self.filenames = self.filenames[:N]
        self.N = len(self.filenames)

        self.dim_ordering = dim_ordering

        self.target_size = target_size

        self.is_a_binary = is_a_binary
        self.is_b_binary = is_b_binary
        self.is_a_grayscale = is_a_grayscale
        self.is_b_grayscale = is_b_grayscale

        self.labels = labels
        self.is_b_categorical = is_b_categorical
        if (self.labels == []):
            self.is_b_categorical = False







        self.channel_index = 3
        self.row_index = 1
        self.col_index = 2

        self.rotation_range = rotation_range
        self.height_shift_range = height_shift_range
        self.width_shift_range = width_shift_range
        self.shear_range = shear_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.sequence_flip = sequence_flip
        self.patches_per_scan = patches_per_scan
        self.separate_output = separate_output
        self.deep_supervision = deep_supervision
        self.weight_map = weight_map

        #here we set the patch dimensions (the same for the loaded scan in case patching = false)
        self.patch_divide = patch_divide
        if self.patch_divide is True:
            self.ptch_sz = ptch_sz
            self.patch_z_sz = patch_z_sz
            self.ptch_str = ptch_str

        else:
            self.patches_per_scan = 1
            self.ptch_sz = self.target_size[0]
            self.patch_z_sz = slice_length

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]

        self.slice_length = slice_length
        if (self.slice_length < 0):
            self.slice_range = [slice_sample, 1]
        else:
            self.slice_range = []

        super(TwoScanIterator, self).__init__(len(self.filenames), batch_size,
                                              shuffle, seed)




    def _binarize(self, batch):
        """Make input binary images have 0 and 1 values only."""
        bin_batch = batch / 255.
        bin_batch[bin_batch >= 0.5] = 1
        bin_batch[bin_batch < 0.5] = 0
        return bin_batch

    def _normalize(self, scan):
        """returns normalized (0 mean 1 variance) scan"""

        scan = (scan - np.mean(scan)) / (np.std(scan))
        return scan


    def load_scan(self, file_name, extension):
        """Load mhd or nrrd 3d scan"""

        if extension == '.mhd':
            scan, _, _ = futil.load_itk(file_name)

        elif extension == '.nrrd':
            scan, _, _ = futil.load_nrrd(file_name)

        return np.expand_dims(scan, axis=-1)

    def _load_img_pair(self, idx,  load_aux=False):
        """Get a pair of images with index idx."""


        a_fname = self.filenames[idx] + self.a_extension
        b_fname = self.filenames[idx] + self.b_extension
        if load_aux:
            c_fname = self.filenames[idx] + self.c_extension

        a = self.load_scan(file_name=os.path.join(self.a_dir, a_fname), extension=self.a_extension)
        b = self.load_scan(file_name=os.path.join(self.b_dir, b_fname), extension=self.b_extension)
        if load_aux:
            c = self.load_scan(file_name=os.path.join(self.c_dir, c_fname), extension=self.c_extension)

        a = np.array(a)
        a = futil.normalize(a)  # we need to change the name of this
        a = self._normalize(a)

        b = np.array(b)

        if load_aux:
            c = np.array(c, dtype='float')

        if (self.slice_length < 0):
            downscale = np.random.uniform(self.slice_range[0], self.slice_range[-1])
            z_length = downscale * a.shape[0]
        else:
            z_length = self.slice_length

        a = self.downscale_scan(a, self.target_size[0], z_length, order=1)
        b = self.downscale_scan(b, self.target_size[0], z_length, order=0)

        if load_aux:
            c = self.downscale_scan(c, self.target_size[0], z_length, order=1)

        if load_aux:
            return a, b, c
        else:
            return a, b

    def downscale_scan(self, scan, sz=128, z_length=64, order=1):

        zoom_seq = np.array([z_length, sz, sz, 1], dtype='float') / np.array(scan.shape, dtype='float')
        s = ndimage.interpolation.zoom(scan, zoom_seq, order=order, prefilter=order)

        return s

    def _one_hot_enc(self, patch, labels=[]):

        labels = np.array(labels)
        N_classes = labels.size

        ptch_ohe = np.zeros((patch.shape[0], patch.shape[0]) + (N_classes,))

        for i, l in enumerate(labels):
            m = np.where((patch == l).all(axis=2))

            new_val = np.zeros(N_classes)
            new_val[i] = 1.

            ptch_ohe[m] = new_val

        return ptch_ohe

    def _random_transform(self, a, b, is_batch=True):
        """
        Random dataset augmentation.

        Adapted from https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
        """

        if is_batch is False:
            # a and b are single images, so they don't have image number at index 0
            img_row_index = self.row_index - 1
            img_col_index = self.col_index - 1
            img_channel_index = self.channel_index - 1
        else:
            img_row_index = self.row_index
            img_col_index = self.col_index
            img_channel_index = self.channel_index
        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range,
                                                    self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) \
                 * a.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) \
                 * a.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix),
                                  zoom_matrix)

        h, w = a.shape[img_row_index], a.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)

        A = []
        B = []

        for a_, b_ in zip(a, b):
            a_ = apply_transform(a_, transform_matrix, img_channel_index - 1,
                                 fill_mode=self.fill_mode, cval=np.min(a_))
            b_ = apply_transform(b_, transform_matrix, img_channel_index - 1,
                                 fill_mode=self.fill_mode, cval=0)

            A.append(a_)
            B.append(b_)

        a = np.array(A)
        b = np.array(B)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                a = flip_axis(a, img_col_index)
                b = flip_axis(b, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                a = flip_axis(a, img_row_index)
                b = flip_axis(b, img_row_index)

        if self.sequence_flip:
            if np.random.random() < 0.5:
                a = flip_axis(a, 0)
                b = flip_axis(b, 0)

        return a, b

    def next(self):
        """Get the next pair of the sequence."""

        # Lock the iterator when the index is changed.
        with self.lock:
            index_array, _, current_batch_size = next(self.index_generator)

        for i, j in enumerate(index_array):
            if self.aux:
                a, b, c = self._load_img_pair(j, self.aux)

                #give weights for the aux map
                if (np.min(c) == 1):
                    c -= 1
                c *= self.weight_map
                c += 1
            else:
                a, b = self._load_img_pair(j,  self.aux)

            if not self.is_b_binary and self.is_b_categorical:
                B_ = []
                #
                b[b >= (np.max(self.labels)+1)] = 0
                for b_ in b:
                    b_ = self._one_hot_enc(b_, self.labels)
                    B_.append(b_)
                b = np.array(B_)

            #we include our aux ground truth in the first channel. This is not cool, but it's easier to make the transformations :(
            if (self.aux):
                np.copyto(b[:, :, :, 0], c.reshape(b[:, :, :, 0].shape))

            # apply random affine transformation
            a, b = self._random_transform(a, b)

            A = []
            B = []
            for _ in range(self.patches_per_scan):

                if self.patch_divide:
                    a_img, b_img = random_patch(a, b, patch_shape=(self.patch_z_sz, self.ptch_sz, self.ptch_sz))
                else:
                    a_img, b_img = a, b

                if (self.aux):
                    b_0 = b_img[:, :, :, 0]
                    b_0[b_0 < 1] = 1

                batch_a = a_img.copy()
                batch_b = b_img.copy()

                if self.is_a_binary:
                    batch_a = self._binarize(batch_a)

                sh = batch_b.shape

                batch_b = np.reshape(batch_b, (sh[0], self.ptch_sz ** 2, sh[self.channel_index]))

                A.append(batch_a)
                B.append(batch_b)

        return [np.array(A), np.array(B)]


    def split_output(self, scan):
        """here we separate again the aux output that was in channel 0"""

        # output 1- original gt
        o1 = scan[np.newaxis, :, :]

        w = scan[:, 0].copy()



        # output 2- weight
        o2 = 1 - (w - 1) / self.weight_map  # background channel
        o2 = o2[:, np.newaxis]
        o2 = np.append(o2, 1 - o2, axis=-1)

        output = [o1, o2[np.newaxis, :, :]]

        for _ in range(self.deep_supervision):
            output.append(o1)

        return output

    def generator(self):

        while 1:
            x, y = self.next()

            # adapt to input tensor
            x_b = np.rollaxis(x, 1, 4)
            y_b = np.rollaxis(y, 1, 3)

            y_b = np.reshape(y_b, (y_b.shape[0], y_b.shape[1] * y_b.shape[2], y_b.shape[-1]))

            for x, y in zip(x_b, y_b):
                if self.separate_output:
                    yield x[np.newaxis, :, :, :, :], self.split_output(y)
                else:
                    yield x[np.newaxis, :, :, :, :], y[np.newaxis, :, :]
