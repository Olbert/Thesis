import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
# read image
file= "E:\Thesis\DomainVis\server_files\processed\data.h5"
with h5py.File(file, 'a') as hf:
	for name in hf.keys():
		im = np.sum(hf[name],axis=0)
		im -= np.min(im)
		im /= np.max(im)
		# calculate mean value from RGB channels and flatten to 1D array
		# vals = im.mean(axis=2).flatten()

		dst = plt.hist(im.ravel(),300,[0,1]) # cv2.calcHist(im, [0],None,100, [0,1])
		# calculate histogram
		# counts, bins = np.histogram(dst, range(256))
		# plot histogram centered on values 0..255
		# plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
		# plt.xlim([-0.5, 255.5])
		plt.savefig(name+'.jpg')
		plt.show()

