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

sub_count = 1
for dirpath, dirnames, filenames in os.walk('./RBDdata/'):
    for filename in filenames:
        if filename.endswith(".edf"):
            subject = dirpath.split('/')[-1]
            print("Processing {}".format(subject))
            raw = mne.io.read_raw_edf(os.path.join(dirpath, filename), verbose=0, preload=True)
            raw = preprocess_raw(raw)
            sampling_rate = raw.info['sfreq']
            label = dirpath.split('/')[-2]
            date = os.path.splitext(filename)[0]

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
