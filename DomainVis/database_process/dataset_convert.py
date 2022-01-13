import gzip
import shutil
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import nibabel as nib
import nibabel.processing
import skimage.transform as skTrans
import h5py
import config
import nilearn
from nilearn.image import resample_img
import matplotlib.pyplot as plt
# Images
zipped_img = "E:\\Thesis\\conp-dataset\\projects\\calgary-campinas\\CC359\\Reconstructed\\Original\\Original"
unzipped_img = "E:\\Thesis\\conp-dataset\\projects\\calgary-campinas\\CC359\\Ready\\images"
npy_img = "E:\\Thesis\\conp-dataset\\projects\\calgary-campinas\\CC359\\Npy\\images"
# Masks
zipped_mask = "E:\\Thesis\\conp-dataset\\projects\\calgary-campinas\\CC359\\" \
              "Reconstructed\\Silver-standard-machine-learning\\Silver-standard"
unzipped_mask = "E:\\Thesis\\conp-dataset\\projects\\calgary-campinas\\CC359\\Ready\\masks"
npy_mask = "E:\\Thesis\\conp-dataset\\projects\\calgary-campinas\\CC359\\Npy\\masks"


def unzip_dataset(zipped, unzipped):
    # Get list of files
    files = [f for f in listdir(zipped) if isfile(join(zipped, f))]

    # Unzip
    for i in range(0, len(files)):
        with gzip.open(zipped + "\\" + files[i], 'rb') as f_in:
            with open(unzipped + "\\" + files[i][:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    # Get list of files
    files_num = len([f for f in listdir(zipped) if isfile(join(zipped, f))])
    print(files_num, " files were unzipped")


def preprocess_dataset(path, new_path, size):
    assert size[0] > 0 and size[1] > 0, 'Scale is too small'

    img_files = [f for f in listdir(path) if isfile(join(path, f))]
    for img_file in img_files:
        pil_img = nib.load(os.path.join(path, img_file)).get_fdata()

        pil_img = np.array(pil_img)
        pil_img = np.clip(np.float32(pil_img), *np.percentile(np.float32(pil_img), [1, 99]))
        pil_img -= np.min(pil_img)
        pil_img /= np.max(pil_img)

        pil_img = skTrans.resize(pil_img, (pil_img.shape[0], size[0], size[1]), order=1, preserve_range=True)
        np.save(os.path.join(new_path, img_file[:-4]), pil_img)


def preprocess_3Dimage(path, new_path="", size=(256, 256)):
    assert size[0] > 0 and size[1] > 0, 'Scale is too small'

    # pil_img = nib.load(path).get_fdata()

    pil_img = nib.load(path).get_fdata()
    pil_img = np.array(pil_img)
    pil_img = np.clip(np.float32(pil_img), *np.percentile(np.float32(pil_img), [1, 99]))
    pil_img -= np.min(pil_img)
    pil_img /= np.max(pil_img)

    pil_img = skTrans.resize(pil_img, (pil_img.shape[0], size[0], size[1]), order=1, preserve_range=True)
    if new_path != "":
        np.save(os.path.join(new_path, path[:-4]), pil_img)
    return pil_img


def preprocess_image(img, size=(256, 256)):
    assert size[0] > 0 and size[1] > 0, 'Scale is too small'

    # pil_img = nib.load(path).get_fdata()

    pil_img = np.array(img)
    pil_img = np.clip(np.float32(pil_img), *np.percentile(np.float32(pil_img), [1, 99]))
    pil_img -= np.min(pil_img)
    pil_img /= np.max(pil_img)


    pil_img = skTrans.resize(pil_img, (pil_img.shape[0], size[0], size[1]), order=1, preserve_range=True)

    return pil_img


def convert_to_h5(dir_in, dir_out, type='data', size=(128, 128), voxel_spacing=(1,1,1),name = None):

    size_set = True
    path = os.path.join(dir_out, type + '.h5')
    if name is None:
        filenames = os.listdir(dir_in)
    else:
        filenames = [name]
    with h5py.File(path, 'a') as hf:
        for filename in filenames:
            img_file = os.path.join(dir_in, filename)
            if os.path.isfile(img_file):

                nib_image = nib.load(img_file)

                out_shape = (nib_image.shape[0],size[0],size[1])
                # preprocessed_dataset = nib.processing.conform(nib_image, voxel_size= voxel_spacing).get_fdata()
                preprocessed_dataset = nibabel.processing.resample_to_output(nib_image, voxel_sizes=[1, 1, 1],mode='nearest').get_fdata()

                img = np.array(preprocessed_dataset)
                if not size_set:
                    size = (img.shape[1], img.shape[2])
                    size_set=True

                img = img[((img.shape[0] - config.SLICES) // 2):((img.shape[0] + config.SLICES) // 2)]

                # downsampled_nii = resample_img(orig_nii, target_affine=np.eye(3) * 2., interpolation='nearest')

                img = preprocess_image(img, size)

                img = (1 * (img - np.min(img)) / np.ptp(img)).astype(float)
                domain = filename.split('.')[0].split('_')[0]# +"_"+filename.split('.')[0].split('_')[2]

                if domain in hf.keys():
                    hf[domain].resize((hf[domain].shape[0] + img.shape[0]), axis=0)
                    hf[domain][-img.shape[0]:] = img
                else:
                    hf.create_dataset(domain, data=img, shape=(config.SLICES, size[0], size[1]), compression="gzip",
                                      chunks=True, maxshape=(None, size[0], size[1]))

            names = [key for key in hf.keys()]
            hf.close()

    return names




def scale_mri(image: np.ndarray) -> np.ndarray:
    image = np.clip(np.float32(image), *np.percentile(np.float32(image), [1, 99]))
    image -= np.min(image)
    image /= np.max(image)
    return np.float32(image)


if __name__ == '__main__':
    # unzip_dataset(zipped_img,unzipped_img)
    # unzip_dataset(zipped_mask, unzipped_mask)

    preprocess_dataset(unzipped_img, npy_img, (256, 256))
    preprocess_dataset(unzipped_mask, npy_mask, (256, 256))
