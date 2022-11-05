import os, re
import numpy as np
import h5py
from tqdm import tqdm
import glob
from sklearn.model_selection import train_test_split
import utils
import pandas as pd
import pprint
from scipy.stats import gmean
import argparse
import yaml
from data import get_label_files
from tqdm import tqdm
import pickle
from model import conv_audio
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description='Code for testing')
parser.add_argument('-model_type', default = 'clean',
                    action='store',
                    required=False,
                    type=str)

parser.add_argument('-alpha',
                    action='store',
                    required=True,
                    type=float)

args = parser.parse_args()
args.alpha/=10

#sc = pickle.load(open('../FSDnoisy18k/features/'+'scaler_clean.pkl','rb'))

path_data =  '../FSDnoisy18k/features/tt_'+args.model_type+'.hdf5'
#path_data =  '../FSDnoisy18k/features/tt_clean.hdf5'
all_files = []
group = []
def func(name, obj):     # function to recursively store all the keys
    if isinstance(obj, h5py.Dataset):
        all_files.append(name)
    elif isinstance(obj, h5py.Group):
        group.append(name)
hf = h5py.File(path_data, 'r')
hf.visititems(func)

# all_data = []
# for i in range(len(all_files)):

#     all_data.append(hf[all_files[i]])

# all_data =np.vstack(np.array(all_data))
# all_data = sc.transform(all_data)

model = conv_audio(20)
model_path = './model_'+ args.model_type +'/model_'+str(args.alpha)+'.pt'
model.load_state_dict(torch.load(model_path))
print(model)



ground_tr_list = []
esti_list=[]
activation_fn = torch.nn.Softmax(dim=1)

for fn in tqdm(all_files):

    ground_truth = int(fn.split('/')[0])
    ground_tr_list.append(ground_truth)


    es_label = torch.zeros(1,20)

    audio = hf[fn]
    # audio = sc.transform(audio)
    #audio = all_data[i*100:i*100+100,:]
    for j in range(audio.shape[0]//100):

        each_data_norm = audio[j*100:j*100+100,:]
        #each_data_norm =  (each_data_norm-np.mean( each_data_norm))/np.std(each_data_norm)

        each_audio_tensor = torch.from_numpy(each_data_norm)[None,None,:,:].float()
        #each_audio_tensor = torch.transpose(each_audio_tensor, 2,3)


        model.eval()
        with torch.no_grad():
            es_label_each_frame,_ = model(each_audio_tensor)

            es_label_each = activation_fn(es_label_each_frame)

        es_label += es_label_each

    es_class = torch.argmax(es_label)
    esti_list.append(es_class)


# for fn in tqdm(all_files):

#     ground_truth = int(fn.split('/')[0])
#     ground_tr_list.append(ground_truth)


#     es_label = torch.zeros(1,20)

#     audio = hf[fn]
#     file_frames = float(audio.shape[0])
#     n_ins = np.maximum(1, int(np.ceil((file_frames -100) / 100)))

#     for j in range(n_ins):
#         data_each_j = audio[j*100:j*100+100,:]
#         audio_tensor = torch.from_numpy(data_each_j)[None,None,:,:].float()
#         model.eval()
#         with torch.no_grad():
#             es_label_each_frame,_ = model(audio_tensor)
#             es_label_each = activation_fn(es_label_each_frame)
#         es_label += es_label_each

#     es_class = torch.argmax(es_label)
#     esti_list.append(es_class)


y_true = np.array(ground_tr_list)
y_pred = np.array(esti_list)
acc = accuracy_score(y_true, y_pred)
acc = np.round(acc,decimals = 3)
print(acc)
