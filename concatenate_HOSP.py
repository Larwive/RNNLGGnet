"""
Some edf files were split.
This is to put them back together.
"""

import mne
import os
from preprocess import preprocess_raw

data_path = './RBDdata/'

to_concat = [
    ['0/07273610/2011-10-24.edf',
     '0/07273610/2011-10-241.edf',
     '0/07273610/2011-10-242.edf'],
    ['0/15551243/2011-11-28.edf',
     '0/15551243/2011-11-281.edf',
     '0/15551243/2011-11-282.edf'],
    ['0/19680557/2011-07-08.edf',
     '0/19680557/2011-07-081.edf'],
    ['0/23484352/2010-09-30.edf',
     '0/23484352/2010-09-301.edf',
     '0/23484352/2010-09-302.edf'],
    ['1/13801845/13801845.edf',
     '1/13801845/138018451.edf']
]

for splitted in to_concat:
    print("Concatenating {}".format(splitted[0]))
    base = mne.io.read_raw_edf(os.path.join(data_path, splitted[0]), preload=True, verbose=0)
    try:
        mne.concatenate_raws([base] + [mne.io.read_raw_edf(os.path.join(data_path, split_path), preload=True, verbose=0) for split_path in splitted[1:]])
    except Exception as e:
        print("Is it already concatenated ?", e)
        continue
    print("New duration: {}s".format(base.n_times/base.info['sfreq']))
    base = preprocess_raw(base)

    try:
        mne.export.export_raw(os.path.join(data_path, splitted[0]), base, overwrite=True, physical_range=(base.get_data().min(), base.get_data().max()), verbose=0)
    except Exception as e:
        print("Failed to export. Trying another way.", e)
        raws = [mne.io.read_raw_edf(os.path.join(data_path, split_path), preload=True, verbose=0) for split_path in splitted]
        raws[0].append(raws[1:], preload=True)
        mne.export.export_raw(os.path.join(data_path, splitted[0]), raws[0], overwrite=True, physical_range=(base.get_data().min(), base.get_data().max()), verbose=0)

    for split_path in splitted[1:]:
        os.remove(os.path.join(data_path, split_path))
