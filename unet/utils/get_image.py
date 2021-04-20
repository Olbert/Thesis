import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
import os
from unet.utils import dataset_convert
from dataset_convert import preprocess_3Dimage
def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


img_path = 'E:\\Thesis\\conp-dataset\\projects\\calgary-campinas\\CC359\\Ready\\images\\CC0140_siemens_15_57_M.nii'
img = preprocess_3Dimage(img_path, size = (128,128))
mask_path = 'E:\\Thesis\\conp-dataset\\projects\\calgary-campinas\\CC359\\Ready\\masks\\CC0140_siemens_15_57_M_ss.nii'
mask = preprocess_3Dimage(mask_path, size = (128,128))

# img = nib.load('E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Ready\images\CC0012_philips_15_43_M.nii')
#
# img = img.get_fdata()

slice_0 = img[42, :, :]
slice_1 = img[:, 70//4, :]
slice_2 = img[:, :, 120//4]

show_slices([slice_0, slice_1, slice_2])
show_slices([mask[42], mask[:,40], mask[:,:,40]])

plt.suptitle("Center slices for EPI image")

plt.show()
mask_slice = mask[42, :, :]
path = os.path.join(os.path.dirname(os.getcwd()),'data')
cv2.imwrite(os.path.join(path, 'mask_s.jpg'), mask_slice* 255)
cv2.imwrite(os.path.join(path, 'input_s.jpg'), slice_0 * 255)

plt.show()



