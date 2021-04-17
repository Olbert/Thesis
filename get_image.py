import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
from unet.utils import dataset_convert
def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


img_path = 'E:\\Thesis\\conp-dataset\\projects\\calgary-campinas\\CC359\\Npy\\images\\CC0012_philips_15_43_M.npy'
img = dataset_convert.preprocess_image(img_path)
# img = nib.load('E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Ready\images\CC0012_philips_15_43_M.nii')
#
# img = img.get_fdata()

slice_0 = img[42, :, :]
slice_1 = img[:, 70, :]
slice_2 = img[:, :, 120]

show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")
cv2.imwrite('image.jpg', slice_0 * 255)
plt.show()



