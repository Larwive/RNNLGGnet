import sys

import mne
import os
import warnings
import pickle
import pandas as pd
import numpy as np
from preprocess import preprocess_raw

warnings.filterwarnings("ignore")

os.makedirs('./RBDdata/dat', exist_ok=True)


def distribute(count0, count1, dict_0, dict_1):
    if count1 > count0:
        count1, count0 = count0, count1
        dict_0, dict_1 = dict_1, dict_0
    rate = count0 // count1 + 1
    while count1 > 0:
        for _ in range(rate):
            if count0 > 0:
                yield dict_0['subjects'].pop(), dict_0['labels'].pop(), dict_0['dates'].pop(), dict_0['dirpaths'].pop(), \
                    dict_0['filenames'].pop()
                count0 -= 1
        yield dict_1['subjects'].pop(), dict_1['labels'].pop(), dict_1['dates'].pop(), dict_1['dirpaths'].pop(), dict_1[
            'filenames'].pop()
        count1 -= 1

dict0 = {
    'subjects': [],
    'labels': [],
    'dates': [],
    'dirpaths': [],
    'filenames': []
}

dict1 = {
    'subjects': [],
    'labels': [],
    'dates': [],
    'dirpaths': [],
    'filenames': []
}

for dirpath, dirnames, filenames in os.walk('./RBDdata/'):
    for filename in filenames:
        if filename.endswith(".edf"):
            subject = dirpath.split('/')[-1]
            label = dirpath.split('/')[-2]
            date = os.path.splitext(filename)[0]
            update_dict = [dict0, dict1][int(label)]
            update_dict['subjects'].append(subject)
            update_dict['labels'].append(label)
            update_dict['dates'].append(date)
            update_dict['dirpaths'].append(dirpath)
            update_dict['filenames'].append(filename)

sub_count = 1
sub_count0 = len(dict0)
sub_count1 = len(dict1)

for subject, label, date, dirpath, filename in distribute(sub_count0, sub_count1, dict0, dict1):
    print("Processing {}".format(subject))
    raw = mne.io.read_raw_edf(os.path.join(dirpath, filename), verbose=0, preload=True)
    raw = preprocess_raw(raw)
    sampling_rate = raw.info['sfreq']

    csvfile = pd.read_csv(os.path.join(dirpath, date + ".csv"))
    mask = csvfile[' Sleep Stage'].str.contains('R')

    groups = mask.ne(mask.shift()).cumsum()

    filtered_blocks = csvfile.loc[mask].groupby(groups).agg(list)
    if len(filtered_blocks['Manual Epoch']) == 0:
        print('No REM for {}'.format(subject), file=sys.stderr)
        continue
    print("REM blocks:")
    for phase, indices in enumerate(filtered_blocks['Manual Epoch']):
        print("{}-{}".format(indices[0], indices[-1]))
        start = int((indices[0] - 1) * sampling_rate * 30)
        end = int(indices[-1] * sampling_rate * 30)
        data_dict = {
            "data": np.array(raw.get_data(start=start, stop=end)),
            "labels": [int(label)],
        }
        new_name = "s{}-{}_{}-{}({}).dat".format(sub_count, subject, date, phase, label)
        with open(os.path.join('./RBDdata/dat/', new_name), 'wb') as f:
            pickle.dump(data_dict, f)
    sub_count += 1
