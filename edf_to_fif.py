import mne
import os
import warnings

warnings.filterwarnings("ignore")


for dirpath, dirnames, filenames in os.walk('./RBDdata/'):
    for filename in filenames:
        if filename.endswith(".edf"):
            raw = mne.io.read_raw_edf(os.path.join(dirpath, filename), verbose=False, preload=True)
            new_name = os.path.splitext(filename)[0] + ".fif"
            raw.save(os.path.join(dirpath, new_name), overwrite=True)