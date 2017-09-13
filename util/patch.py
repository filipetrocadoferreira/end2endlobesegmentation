import numpy as np
import scipy.stats as st
from keras import backend as K
import matplotlib.pyplot as plt
class Patches(object):

    def __init__(self, sz_patch=[128,128,64], stride=None, inFov=True, mode='exclude_border'):

        self.sz_patch = sz_patch
        self.inFov = inFov
        if inFov is True:
            self.mode = mode

        self.dim_ordering = K.image_dim_ordering()
        # if stride = None, there is no overlap
        if stride is None:
            self.stride = self.sz_patch
        else:
            self.stride = stride

    def _inside_FOV(self, patch):

        if self.mode is 'exclude_border':
            if np.all((patch == [0, 0, 0]).all(axis=2)):
                r = False
            else:
                r = True
        elif self.mode is 'exclude_any':
            if np.any((patch == [0, 0, 0]).all(axis=2)):
                r = False
            else:
                r = True
        elif self.mode is 'triclass':
            if np.max(patch) > 1:
                px = 255
            else:
                px = 1
            if np.any((patch == [px,px,px]).all(axis=2)):
                r = False
            elif np.any((patch == [px, px, 0]).all(axis=2)):
                r = False
            elif (np.any((patch == [px, 0, 0]).all(axis=2)) and
                  np.any((patch == [0, px, 0]).all(axis=2)) and
                  np.any((patch == [0, 0, px]).all(axis=2))):
                r = True
            else:
                r = False

        elif self.mode is 'triclass_intersect':
            if np.max(patch) > 1:
                px = 255
            else:
                px = 1
            if np.any((patch == [px, px, 0]).all(axis=2)):
                g = np.where((patch == [px, px, 0]).all(axis=2))
                patch[g] = [0,0,px]

            if np.any((patch == [px,px,px]).all(axis=2)):
                r = False
            elif (np.any((patch == [px, 0, 0]).all(axis=2)) and
                  np.any((patch == [0, px, 0]).all(axis=2)) and
                  np.any((patch == [0, 0, px]).all(axis=2))):
                r = True
            else:
                r = False
                #print r

            #print r
        else:
            raise Exception(
                'mode is not valid. If inFov is set to True mode can be one of:'
                '"exclude_border", "exclude_any", "triclass" or "triclass_intersect".'
                'Got {0}'.format(self.mode))
        return r

    def create_patches(self, img, gdt=None, mask=None):
    # create patches from the prepocessed images
        if self.dim_ordering != 'tf':
            if len(img.shape) < 3:
                img = img
            else:
                img = np.transpose(img, (1, 2, 0))

        if len(img.shape) < 3:
            img = img[:, :, np.newaxis]
        if mask is None:
            masked_img = img
        else:
            if len(mask.shape) < 3:
                masked_img = img * (mask[:, :, np.newaxis])
            else:
                masked_img = img * mask

        numpatch_h = int((masked_img.shape[0] - self.sz_patch[0] + self.stride)
                         / self.stride)
        numpatch_w = int((masked_img.shape[1] - self.sz_patch[1] + self.stride)
                         / self.stride)

        # Make sure that the whole image is
        if (masked_img.shape[0] - self.sz_patch[0] + self.stride) % self.stride > 0:
            last_h = True
            numpatch_h = numpatch_h + 1
        else:
            last_h = False
        if (int((masked_img.shape[1] - self.sz_patch[1] + self.stride) % self.stride)) > 0:
            last_w = True
            numpatch_w = numpatch_w + 1
        else:
            last_w = False

        img_patch = []
        Ind_reco = []
        #

        if gdt is not None:
            if len(gdt.shape) < 3:
                gdt = gdt[:, :, np.newaxis]
                gdt_patch = []
            else:
                if self.dim_ordering != 'tf':
                    gdt = np.transpose(gdt, (1, 2, 0))
                gdt_patch = []

        for nh in range(0,numpatch_h):

            if last_h is True and nh == numpatch_h-1:
                end_h = img.shape[0]
                start_h = img.shape[0] - self.sz_patch[0]
            else:
                start_h = int(nh * self.stride)
                end_h = int(start_h + self.sz_patch[0])


            for nw in range(0,numpatch_w):
                if last_w is True and nw == numpatch_w-1:
                    end_w = masked_img.shape[1]
                    start_w = masked_img.shape[1] - self.sz_patch[1]
                else:
                    start_w = int(nw * self.stride)
                    end_w = int(start_w + self.sz_patch[1])
                patch = masked_img[start_h:end_h, start_w:end_w, :]
                if self.inFov is True:
                    if ((self.mode is 'triclass' and gdt is None) or
                            (self.mode is 'triclass_intersect' and gdt is None)):
                        raise Exception('Mode "triclass" is only valid when'
                                        ' a Ground Truth is provided.')
                    elif ((self.mode is 'triclass' and gdt is not None) or
                            (self.mode is 'triclass_intersect' and gdt is not None)):

                        gdt_p = gdt[start_h:end_h, start_w:end_w, :]
                        if self._inside_FOV(gdt_p) is True:
                            img_patch.append(patch)
                            Ind_reco.append((start_h, start_w))
                            gdt_patch.append(gdt_p)
                        else:
                            pass

                    else:
                        if self._inside_FOV(patch) is True:
                            img_patch.append(patch)
                            Ind_reco.append((start_h, start_w))
                            if gdt is not None:
                                gdt_patch.append(
                                    gdt[start_h:end_h, start_w:end_w, :])

                else:
                    img_patch.append(patch)
                    Ind_reco.append((start_h, start_w))
                    if gdt is not None:
                        gdt_patch.append(gdt[start_h:end_h, start_w:end_w, :])

        img_patch_ar = np.array(img_patch)
        if gdt is not None:
            gdt_patch_ar = np.array(gdt_patch)

        if self.dim_ordering != 'tf':
            img_patch_ar = np.transpose(img_patch_ar, (0, 3, 2, 1))
            if gdt is not None:
                gdt_patch_ar = np.transpose(gdt_patch_ar, (0, 3, 2, 1))


        if gdt is not None:
            return img_patch_ar, Ind_reco, gdt_patch_ar
        else:
            return img_patch_ar, Ind_reco



    def reconstruct_img(self, img_patch, img_shape, Ind_reco=None):
        """
        Reconstruct the images
        mode: "average" or "absolute"
        """
        if self.dim_ordering != 'tf':
            if len(img_patch.shape) < 4:
                img_patch = img_patch
            else:
                img_patch = np.transpose(img_patch, (0, 2, 3, 1))

        if len(img_patch.shape) < 4:
            img_patch = img_patch[:, :, :, np.newaxis]
            nch = 1
        else:
            nch = img_patch.shape[-1]

        new_img = np.zeros(img_shape)
        weights = np.zeros(img_shape)
        if nch == 1 and len(img_shape) < 3:
            new_img = new_img[:, :, np.newaxis]
            weights = weights[:, :, np.newaxis]

        npatch = 0
        if Ind_reco is None:
            numpatch_h = int((img_shape[0] - self.sz_patch[0] + self.stride) / self.stride)
            numpatch_w = int((img_shape[1] - self.sz_patch[0] + self.stride) / self.stride)
            for nh in range(numpatch_h):
                start_h = int(nh * self.stride)
                end_h = int(start_h + self.sz_patch[0])

                for nw in range(numpatch_w):
                    start_w = int(nw * self.stride)
                    end_w = int(start_w + self.sz_patch[1])

                    if self.mode is 'average':
                        new_img[start_h:end_h, start_w:end_w, :] += img_patch[npatch]
                        weights[start_h:end_h, start_w:end_w, :] += 1
                    else:
                        new_img[start_h:end_h, start_w:end_w, :] = img_patch[npatch]
                        weights[start_h:end_h, start_w:end_w, :] = 1
                        #
                    npatch += 1
        else:
            kernel = gkern(kernlen=self.sz_patch[0])[:,:,np.newaxis]
            for ind in range(len(Ind_reco)):

                start_h = Ind_reco[ind][0]
                end_h = int(start_h + self.sz_patch[0])
                start_w = Ind_reco[ind][1]
                end_w = int(start_w + self.sz_patch[1])
                if self.mode is 'average':
                    new_img[start_h:end_h, start_w:end_w, :] += img_patch[npatch]*kernel
                    weights[start_h:end_h, start_w:end_w, :] += kernel
                else:
                    new_img[start_h:end_h, start_w:end_w, :] = img_patch[npatch]*kernel
                    weights[start_h:end_h, start_w:end_w, :] = kernel

                # weights[start_h:end_h, start_w:end_w, :] += 1

                npatch += 1

        weights[weights == 0] = 1
        reconst_img = new_img / weights

        return reconst_img

def gkern(kernlen=32, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.) / (kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

