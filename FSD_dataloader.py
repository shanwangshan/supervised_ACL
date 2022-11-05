
import torch
from torch.utils import data
import numpy as np
import os
import os.path
from os import path

import h5py
#from IPython import embed
import pickle
from sklearn.preprocessing import StandardScaler
'''
This script is part of the train.py, it defines the dataset as a class
'''
class FSD18(data.Dataset):

    'Characterizes a dataset for PyTorch'
    def __init__(self, data_ty, path_features):
        super(FSD18, self).__init__()
        self.data_ty = data_ty
        self.path_features = path_features

        if 'tr' in self.data_ty:
            self.path_input = self.path_features+self.data_ty+'.hdf5'
            print(self.path_input)

        if 'val' in self.data_ty:
            self.path_input =self.path_features+self.data_ty+'.hdf5'
            print(self.path_input)

        self.all_files = []
        self.group = []
        def func(name, obj):
            if isinstance(obj, h5py.Dataset):
                self.all_files.append(name)
            elif isinstance(obj, h5py.Group):
                self.group.append(name)
        self.hf = h5py.File(self.path_input, 'r')
        self.hf.visititems(func)
        self.hf.close()


        print('loading is done')

    def __len__(self):
            'Denotes the total number of samples'
            #print('total number of files is', len(self.all_files))
            return len(self.all_files)

    def __getitem__(self,index):

        #log_mel =self.sc.transform(np.array(self.hf[self.all_files[index]]))
        hf = h5py.File(self.path_input, 'r')

        log_mel =np.array(hf[self.all_files[index]])
        # log_mel_norm = (log_mel-np.mean(log_mel))/np.std(log_mel)
        #log_mel_norm = self.sc.transform(log_mel)
        #print(log_mel_norm)

        #log_mel_norm = self.all_data[index*100:index*100+100,:]


        #print(log_mel)#,np.mean(log_mel[5,:], np.std(log_mel[5,:])))

        ground_tr = np.array(int(float(self.all_files[index].split('/')[0])))
        #print(self.all_files[index],ground_tr)

        normed_embed_tensor = torch.from_numpy(log_mel).float()
        normed_embed_tensor =  normed_embed_tensor[None, :,:]
        #normed_embed_tensor = torch.transpose(normed_embed_tensor, 1, 2)
        ground_tr_tensor=torch.from_numpy(ground_tr).long()

        return normed_embed_tensor,ground_tr_tensor
