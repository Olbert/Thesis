import imageio
import numpy as np
import h5py

# f = h5py.File('E:\\Thesis\\gdrive\\test\\ge_3\\data.h5', 'r')
# dset = f['ge_3']
# data = np.array(dset[5,:,:])
# file = 'test.png' # or .jpg
# imageio.imwrite(file, data)


file= "E:\\Thesis\\DomainVis\\server_files\\processed\\full_data.h5"
with h5py.File(file, 'a') as hf:
	for name in hf.keys():
		im = hf[name][100]

		imageio.imwrite('images\\'+ name+'_brain1.jpg', im)
