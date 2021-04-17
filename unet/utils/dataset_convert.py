import gzip
import shutil
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import nibabel as nib
import skimage.transform as skTrans



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

        pil_img = skTrans.resize(pil_img, (pil_img.shape[0], size[0],size[1]), order=1, preserve_range=True)
        np.save(os.path.join(new_path, img_file[:-4]),pil_img)

def preprocess_image(path, new_path = "", size = (256,256)):
    assert size[0] > 0 and size[1] > 0, 'Scale is too small'

    #pil_img = nib.load(path).get_fdata()
    pil_img = np.load(path)
    pil_img = np.array(pil_img)
    pil_img = np.clip(np.float32(pil_img), *np.percentile(np.float32(pil_img), [1, 99]))
    pil_img -= np.min(pil_img)
    pil_img /= np.max(pil_img)

    pil_img = skTrans.resize(pil_img, (pil_img.shape[0], size[0],size[1]), order=1, preserve_range=True)
    if new_path != "":
        np.save(os.path.join(new_path, path[:-4]), pil_img)
    return pil_img

if __name__ == '__main__':
    # unzip_dataset(zipped_img,unzipped_img)
    # unzip_dataset(zipped_mask, unzipped_mask)


    preprocess_dataset(unzipped_img, npy_img, (256,256))
    preprocess_dataset(unzipped_mask, npy_mask, (256,256))