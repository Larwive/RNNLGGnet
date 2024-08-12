import sys
import mne
import os
import warnings
import pickle
import pandas as pd
import numpy as np
from preprocess import preprocess_raw
from utils import print_cyan

warnings.filterwarnings("ignore")

# Rem
datpath = 'RBDdat'
label0path = '2650142/DatabaseSubjects/'
label1path = 'RBDdataPark/'
kept_channels = ['ECG EKG', 'EEG O1-A2', 'EEG O2-A1']

""" # Parkinson
datpath = 'RBDdataPark/dat/'
label0path = 'RBDdataPark/0/'
label1path = 'RBDdataPark/1/'
kept_channels = None
"""

os.makedirs(datpath, exist_ok=True)


def distribute(count0, count1, dict_0, dict_1):
    if count1 > count0:
        count1, count0 = count0, count1
        dict_0, dict_1 = dict_1, dict_0
    rate = count0 // count1  # + 1  # Adjust if needed
    while count1 > 0 or count0 > 0:
        for _ in range(rate):
            if count0 > 0:
                yield dict_0['subjects'].pop(), dict_0['labels'].pop(), dict_0['dates'].pop(), dict_0['dirpaths'].pop(), \
                    dict_0['filenames'].pop(), dict_0['stagetime'].pop()
                count0 -= 1
        yield dict_1['subjects'].pop(), dict_1['labels'].pop(), dict_1['dates'].pop(), dict_1['dirpaths'].pop(), dict_1[
            'filenames'].pop(), dict_1['stagetime'].pop()
        count1 -= 1


dict0 = {
    'subjects': [],
    'labels': [],
    'dates': [],
    'dirpaths': [],
    'filenames': [],
    'stagetime': []
}

dict1 = {
    'subjects': [],
    'labels': [],
    'dates': [],
    'dirpaths': [],
    'filenames': [],
    'stagetime': []
}

for label, datapath, update_dict, stagetime in zip(['0', '1'], [label0path, label1path], [dict0, dict1], [5, 30]):
    for dirpath, dirnames, filenames in os.walk(datapath):
        for filename in filenames:
            if filename.endswith(".edf"):
                if stagetime == 30:
                    subject = dirpath.split('/')[-1]
                else:
                    subject = int(filename[7:-4])
                date = os.path.splitext(filename)[0]
                update_dict['subjects'].append(subject)
                update_dict['labels'].append(label)
                update_dict['dates'].append(date)
                update_dict['dirpaths'].append(dirpath)
                update_dict['filenames'].append(filename)
                update_dict['stagetime'].append(stagetime)

sub_count = 1
sub_count0 = len(dict0['subjects'])
sub_count1 = len(dict1['subjects'])

for subject, label, date, dirpath, filename, stagetime in distribute(sub_count0, sub_count1, dict0, dict1):
    print_cyan("Processing {} ({})".format(subject, sub_count))
    raw = mne.io.read_raw_edf(os.path.join(dirpath, filename), verbose=0, preload=True)
    raw = preprocess_raw(raw, kept_channels=kept_channels)
    sampling_rate = raw.info['sfreq']

    if stagetime == 30:  # Lazy differentiation
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
            start = int((indices[0] - 1) * sampling_rate * stagetime)
            end = int(indices[-1] * sampling_rate * stagetime)
            data_dict = {
                "data": np.array(raw.get_data(start=start, stop=end)),
                "labels": [int(label)],
            }
            new_name = "s{}-{}_{}-{}({}).dat".format(sub_count, subject, date, phase, label)
            with open(os.path.join(datpath, new_name), 'wb') as f:
                pickle.dump(data_dict, f)
    else:  # stagetime=5
        with open(os.path.join(dirpath, "HypnogramAASM_{}.txt".format(filename.split('.')[0])), 'r') as f:
            lines = f.readlines()
            start_time, current, phase = None, 1, 0
            for line in lines[1:]:
                current += 1
                if start_time is None:
                    if int(line) == 4:
                        start_time = current
                else:
                    if int(line) != 4:
                        print("{}-{}".format(start_time, current - 1))
                        start = int(start_time * sampling_rate * stagetime)
                        end = int((current - 1) * sampling_rate * stagetime)
                        data_dict = {
                            "data": np.array(raw.get_data(start=start, stop=end)),
                            "labels": [int(label)],
                        }
                        new_name = "s{}-{}_{}-{}({}).dat".format(sub_count, subject, date, phase, label)
                        with open(os.path.join(datpath, new_name), 'wb') as f:
                            pickle.dump(data_dict, f)
                        start_time = None
                        phase = 1

    sub_count += 1
