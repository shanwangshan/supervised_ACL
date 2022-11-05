
import numpy as np
import os
import utils
from sklearn.preprocessing import StandardScaler


# NOTE:
# these data generators work for small-medium size datasets under no memory constraints, eg RAM 32GB or more.
# If used with smaller RAMs, a slightly different approach for feeding the net may be needed.


def get_label_files(filelist=None, dire=None, suffix_in=None, suffix_out=None):
    """

    :param filelist:
    :param dire:
    :param suffix_in:
    :param suffix_out:
    :return:
    """

    nb_files_total = len(filelist)
    labels = np.zeros((nb_files_total, 1), dtype=np.float32)
    for f_id in range(nb_files_total):
        labels[f_id] = utils.load_tensor(in_path=os.path.join(dire, filelist[f_id].replace(suffix_in, suffix_out)))
    return labels
