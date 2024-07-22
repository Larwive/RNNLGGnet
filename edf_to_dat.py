import mne
import os
import warnings
from preprocess import preprocess_raw
import pickle

warnings.filterwarnings("ignore")

for dirpath, dirnames, filenames in os.walk('./RBDdata/'):
    for filename in filenames:
        if filename.endswith(".edf"):
            raw = mne.io.read_raw_edf(os.path.join(dirpath, filename), verbose=False, preload=True)
            data, labels = preprocess_raw(raw)
            new_name = os.path.splitext(filename)[0] + ".dat"
            with open(os.path.join(dirpath, new_name), 'wb') as f:
                pickle.dump(data, f)
