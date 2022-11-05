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
from sklearn.preprocessing import StandardScaler
from feat_ext import load_audio_file, get_mel_spectrogram, modify_file_variable_length
parser = argparse.ArgumentParser(description='supervised learning code for ICASSP2022 paper Self-supervised learning of audio representations of using angular contrastive loss')
parser.add_argument('-p', '--params_yaml',
                    dest='params_yaml',
                    action='store',
                    required=False,
                    type=str)
args = parser.parse_args()
print('\nYaml file with parameters defining the experiment: %s\n' % str(args.params_yaml))


params = yaml.full_load(open(args.params_yaml))
params_ctrl = params['ctrl']
params_extract = params['extract']
params_learn = params['learn']
params_loss = params['loss']
params_recog = params['recognizer']

suffix_in = params['suffix'].get('in')
suffix_out = params['suffix'].get('out')


params_extract['audio_len_samples'] = int(params_extract.get('fs') * params_extract.get('audio_len_s'))


# ======================================================== PATHS FOR DATA, FEATURES and GROUND TRUTH
# where to look for the dataset
path_root_data = params_ctrl.get('dataset_path')

params_path = {'path_to_features': os.path.join(path_root_data, 'features'),
               'featuredir_tr': 'audio_train_varup2/',
               'featuredir_te': 'audio_test_varup2/',
               'path_to_dataset': path_root_data,
               'audiodir_tr': 'FSDnoisy18k.audio_train/',
               'audiodir_te': 'FSDnoisy18k.audio_test/',
               'audio_shapedir_tr': 'audio_train_shapes/',
               'audio_shapedir_te': 'audio_test_shapes/',
               'gt_files': os.path.join(path_root_data, 'FSDnoisy18k.meta')}


params_path['featurepath_tr'] = os.path.join(params_path.get('path_to_features'), params_path.get('featuredir_tr'))
params_path['featurepath_te'] = os.path.join(params_path.get('path_to_features'), params_path.get('featuredir_te'))

params_path['audiopath_tr'] = os.path.join(params_path.get('path_to_dataset'), params_path.get('audiodir_tr'))
params_path['audiopath_te'] = os.path.join(params_path.get('path_to_dataset'), params_path.get('audiodir_te'))

params_path['audio_shapepath_tr'] = os.path.join(params_path.get('path_to_dataset'),
                                                 params_path.get('audio_shapedir_tr'))
params_path['audio_shapepath_te'] = os.path.join(params_path.get('path_to_dataset'),
                                                 params_path.get('audio_shapedir_te'))


# ======================================================== SPECIFIC PATHS TO SOME IMPORTANT FILES
# ground truth, load model, save model, predictions, results
params_files = {'gt_test': os.path.join(params_path.get('gt_files'), 'test.csv'),
                'gt_train': os.path.join(params_path.get('gt_files'), 'train.csv')}

# # ============================================= print all params to keep record in output file
print('\nparams_ctrl=')
pprint.pprint(params_ctrl, width=1, indent=4)
print('params_files=')
pprint.pprint(params_files, width=1, indent=4)
print('params_extract=')
pprint.pprint(params_extract, width=1, indent=4)
print('params_learn=')
pprint.pprint(params_learn, width=1, indent=4)
print('params_loss=')
pprint.pprint(params_loss, width=1, indent=4)
print('params_recog=')
pprint.pprint(params_recog, width=1, indent=4)
print('\n')


# ============================================================== READ TRAIN and TEST DATA
# ============================================================== READ TRAIN and TEST DATA
# ============================================================== READ TRAIN and TEST DATA
# ============================================================== READ TRAIN and TEST DATA

# aim: lists with all wav files for tr and te
train_csv = pd.read_csv(params_files.get('gt_train'))
test_csv = pd.read_csv(params_files.get('gt_test'))
filelist_audio_tr = train_csv.fname.values.tolist()
filelist_audio_te = test_csv.fname.values.tolist()

# get positions of manually_verified clips: separate between CLEAN and NOISY sets
filelist_audio_tr_flagveri = train_csv.manually_verified.values.tolist()
idx_flagveri = [i for i, x in enumerate(filelist_audio_tr_flagveri) if x == 1]
idx_flagnonveri = [i for i, x in enumerate(filelist_audio_tr_flagveri) if x == 0]

# create list of ids that come from the noisy set
noisy_ids = [int(filelist_audio_tr[i].split('.')[0]) for i in idx_flagnonveri]
params_learn['noisy_ids'] = noisy_ids

# get positions of clips of noisy_small subset
# subset of the NOISY set of comparable size to that of CLEAN
filelist_audio_tr_nV_small_dur = train_csv.noisy_small.values.tolist()
idx_nV_small_dur = [i for i, x in enumerate(filelist_audio_tr_nV_small_dur) if x == 1]

# create dict with ground truth mapping with labels:
# -key: path to wav
# -value: the ground truth label too
file_to_label = {params_path.get('audiopath_tr') + k: v for k, v in
                 zip(train_csv.fname.values, train_csv.label.values)}

# ========================================================== CREATE VARS FOR DATASET MANAGEMENT
# list with unique n_classes labels and aso_ids
list_labels = sorted(list(set(train_csv.label.values)))
list_aso_ids = sorted(list(set(train_csv.aso_id.values)))

# create dicts such that key: value is as follows
# label: int
# int: label
label_to_int = {k: v for v, k in enumerate(list_labels)}
int_to_label = {v: k for k, v in label_to_int.items()}

# create ground truth mapping with categorical values
file_to_int = {k: label_to_int[v] for k, v in file_to_label.items()}

if params_ctrl.get('feat_ext'):
    n_extracted_tr = 0; n_extracted_te = 0; n_failed_tr = 0; n_failed_te = 0

    # only if features have not been extracted, ie
    # if folder does not exist, or it exists with less than 80% of the feature files
    # create folder and extract features
    nb_files_tr = len(filelist_audio_tr)
    if not os.path.exists(params_path.get('featurepath_tr')) or \
                    len(os.listdir(params_path.get('featurepath_tr'))) < nb_files_tr*0.8:
        os.makedirs(params_path.get('featurepath_tr'))
        os.makedirs(params_path.get('featurepath_te'))

        for idx, f_name in enumerate(filelist_audio_tr):
            f_path = os.path.join(params_path.get('audiopath_tr'), f_name)
            if os.path.isfile(f_path) and f_name.endswith('.wav'):
                # load entire audio file and modify variable length, if needed
                y = load_audio_file(f_path, input_fixed_length=params_extract['audio_len_samples'], params_extract=params_extract)
                y = modify_file_variable_length(data=y,
                                                input_fixed_length=params_extract['audio_len_samples'],
                                                params_extract=params_extract)

                # compute log-scaled mel spec. row x col = time x freq
                # this is done only for the length specified by loading mode (fix, varup, varfull)
                mel_spectrogram = get_mel_spectrogram(audio=y, params_extract=params_extract)

                # save the T_F rep to a binary file (only the considered length)
                utils.save_tensor(var=mel_spectrogram,
                                  out_path=os.path.join(params_path.get('featurepath_tr'),
                                                        f_name.replace('.wav', '.data')), suffix='_mel')

                # save also label
                utils.save_tensor(var=np.array([file_to_int[f_path]], dtype=float),
                                  out_path=os.path.join(params_path.get('featurepath_tr'),
                                                        f_name.replace('.wav', '.data')), suffix='_label')

                if os.path.isfile(os.path.join(params_path.get('featurepath_tr'),
                                               f_name.replace('.wav', suffix_in + '.data'))):
                    n_extracted_tr += 1
                    print('%-22s: [%d/%d] of %s' % ('Extracted tr features', (idx + 1), nb_files_tr, f_path))
                else:
                    n_failed_tr += 1
                    print('%-22s: [%d/%d] of %s' % ('FAILING to extract tr features', (idx + 1), nb_files_tr, f_path))
            else:
                print('%-22s: [%d/%d] of %s' % ('this tr audio is in the csv but not in the folder', (idx + 1), nb_files_tr, f_path))

        print('n_extracted_tr: {0} / {1}'.format(n_extracted_tr, nb_files_tr))
        print('n_failed_tr: {0} / {1}\n'.format(n_failed_tr, nb_files_tr))

        nb_files_te = len(filelist_audio_te)
        for idx, f_name in enumerate(filelist_audio_te):
            f_path = os.path.join(params_path.get('audiopath_te'), f_name)
            if os.path.isfile(f_path) and f_name.endswith('.wav'):
                # load entire audio file and modify variable length, if needed
                y = load_audio_file(f_path, input_fixed_length=params_extract['audio_len_samples'], params_extract=params_extract)
                y = modify_file_variable_length(data=y,
                                                input_fixed_length=params_extract['audio_len_samples'],
                                                params_extract=params_extract)

                # compute log-scaled mel spec. row x col = time x freq
                # this is done only for the length specified by loading mode (fix, varup, varfull)
                mel_spectrogram = get_mel_spectrogram(audio=y, params_extract=params_extract)

                # save the T_F rep to a binary file (only the considered length)
                utils.save_tensor(var=mel_spectrogram,
                                  out_path=os.path.join(params_path.get('featurepath_te'),
                                                          f_name.replace('.wav', '.data')), suffix='_mel')

                if os.path.isfile(os.path.join(params_path.get('featurepath_te'),
                                               f_name.replace('.wav', '_mel.data'))):
                    n_extracted_te += 1
                    print('%-22s: [%d/%d] of %s' % ('Extracted te features', (idx + 1), nb_files_te, f_path))
                else:
                    n_failed_te += 1
                    print('%-22s: [%d/%d] of %s' % ('FAILING to extract te features', (idx + 1), nb_files_te, f_path))
            else:
                print('%-22s: [%d/%d] of %s' % ('this te audio is in the csv but not in the folder', (idx + 1), nb_files_te, f_path))

        print('n_extracted_te: {0} / {1}'.format(n_extracted_te, nb_files_te))
        print('n_failed_te: {0} / {1}\n'.format(n_failed_te, nb_files_te))
#
# ============================================================BATCH GENERATION
# ============================================================BATCH GENERATION
# ============================================================BATCH GENERATION

# Assuming features or T-F representations on a per-file fashion previously computed and in disk
# input: '_mel'
# output: '_label'

# select the subset of training data to consider: all, clean, noisy, noisy_small_dur
if params_ctrl.get('train_data') == 'all':
    ff_list_tr = [f for f in os.listdir(params_path.get('featurepath_tr')) if f.endswith(suffix_in + '.data') and
                  os.path.isfile(os.path.join(params_path.get('featurepath_tr'), f.replace(suffix_in, suffix_out)))]

elif params_ctrl.get('train_data') == 'clean':
    # only files (not path), feature file list for tr, only those that are manually verified: CLEAN SET
    ff_list_tr = [filelist_audio_tr[i].replace('.wav', suffix_in + '.data') for i in idx_flagveri]

elif params_ctrl.get('train_data') == 'noisy':
    # only files (not path), feature file list for tr, only those that are NOT verified: NOISY SET
    ff_list_tr = [filelist_audio_tr[i].replace('.wav', suffix_in + '.data') for i in idx_flagnonveri]

elif params_ctrl.get('train_data') == 'noisy_small':
    # only files (not path), feature file list for tr, only a small portion of the NOISY SET
    # (comparable to CLEAN SET in terms of duration)
    ff_list_tr = [filelist_audio_tr[i].replace('.wav', suffix_in + '.data') for i in idx_nV_small_dur]

# get label for every file *from the .data saved in disk*, in float
labels_audio_train = get_label_files(filelist=ff_list_tr,
                                     dire=params_path.get('featurepath_tr'),
                                     suffix_in=suffix_in,
                                     suffix_out=suffix_out
                                     )

# sanity check
print('Number of clips considered as train set: {0}'.format(len(ff_list_tr)))
print('Number of labels loaded for train set: {0}'.format(len(labels_audio_train)))

# split the val set randomly (but stratified) within the train set
tr_files, val_files = train_test_split(ff_list_tr,
                                       test_size=params_learn.get('val_split'),
                                       stratify=labels_audio_train,
                                       random_state=42
                                       )






data_all = []
for fn in tqdm(tr_files):
    data_each = utils.load_tensor(params_path.get('featurepath_tr')+fn)

    file_frames = float(data_each.shape[0])

    n_ins = np.maximum(1, int(np.ceil((file_frames -100) / 50)))

    for j in range(n_ins):

        data_each_j = data_each[j*50:j*50+100,:]

        data_all.append(data_each_j)

data_array = np.vstack(np.array(data_all))
print(data_array.shape)
scaler = StandardScaler()
scaler.fit(data_array)



save_data = '../FSDnoisy18k/features/'

hf_tr = h5py.File(save_data+'tr_'+params_ctrl.get('train_data')+'.hdf5', 'a')

for fn in tqdm(tr_files):

    data_tr = utils.load_tensor(params_path.get('featurepath_tr')+fn)
    data_norm_tr = scaler.transform(data_tr)


    labels_fn = fn.replace('mel','label')
    label = int(utils.load_tensor(params_path.get('featurepath_tr')+labels_fn))

    file_frames = float(data_norm_tr.shape[0])
    n_ins = np.maximum(1, int(np.ceil((file_frames -100) / 50)))

    grp_audio_tr = hf_tr.create_group(str(label)+'/'+fn.split('.')[0])
    for j in range(n_ins):
        data_each_j = data_norm_tr[j*50:j*50+100,:]
        grp_audio_tr.create_dataset(str(j), data=data_each_j)

hf_tr.close()

# generate validation files #####

hf_val = h5py.File(save_data+'val_'+params_ctrl.get('train_data')+'.hdf5', 'a')
for fn in tqdm(val_files):

    data_val = utils.load_tensor(params_path.get('featurepath_tr')+fn)
    data_norm_val = scaler.transform(data_val)

    labels_fn = fn.replace('mel','label')
    label = int(utils.load_tensor(params_path.get('featurepath_tr')+labels_fn))

    file_frames = float(data_norm_val.shape[0])
    n_ins = np.maximum(1, int(np.ceil((file_frames -100) / 50)))

    grp_audio_val = hf_val.create_group(str(label)+'/'+fn.split('.')[0])
    for j in range(n_ins):
        data_each_j = data_norm_val[j*50:j*50+100,:]
        grp_audio_val.create_dataset(str(j), data=data_each_j)


hf_val.close()


hf_tt = h5py.File(save_data+'tt_'+params_ctrl.get('train_data')+'.hdf5', 'a')
for fn in tqdm(filelist_audio_te):

    audio_data_tt = utils.load_tensor(params_path.get('featurepath_te')+fn.replace('.wav','_mel.data'))
    data_norm_tt = scaler.transform(audio_data_tt)


    label = test_csv.loc[test_csv['fname']==fn]['label'].values[0]
    label = int(label_to_int[label])


    hf_tt.create_dataset(str(label)+'/'+fn, data=data_norm_tt)


hf_tt.close()
