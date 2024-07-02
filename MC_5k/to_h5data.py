import numpy as np
import h5py

#path_a = "/scratch/project_2009916/PUGAN-Pytorch/MC_5k/Mydataset/PUNET/uniform/train/INPUT/armadillo.xyz"
#path_b = "/scratch/project_2009916/PUGAN-Pytorch/MC_5k/Mydataset/PUNET/uniform/train/INPUT/big_girl.xyz"

#np_a = np.loadtxt(path_a)
#np_b = np.loadtxt(path_b)

#h5f = h5py.File('data.h5','w')
#h5f.create_dataset('poisson_uniform_1024', data=[np_a, np_b])
#print(h5f['poisson_uniform_1024'])
#h5f.close

h5_file_path = 'Mydataset/PUNET/traingt_non_uniform_4096.h5'
h5_file = h5py.File(h5_file_path)
print('XXX ',h5_file.keys())
print('XXX ',h5_file['gt_non_uniform_4096'][:].shape)

