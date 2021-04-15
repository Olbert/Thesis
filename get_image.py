import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cv2

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

img = nib.load('E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Ready\masks\CC0002_philips_15_56_M_ss.nii')

img = img.get_fdata()
slice_0 = img[42, :, :]
slice_1 = img[:, 70, :]
slice_2 = img[:, :, 120]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")
cv2.imwrite('mask.jpg', slice_0)
plt.show()



