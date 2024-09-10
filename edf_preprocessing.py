import mne
import os
import warnings
from preprocess import preprocess_raw
from utils import print_cyan

warnings.filterwarnings("ignore")

# Rem
datpath = 'stw2/'
#label0path = '2650142/DatabaseSubjects/'
label1path = 'RBDdataPark/'
kept_channels = None  # ['ECG EKG', 'EEG O1-A2', 'EEG O2-A1', 'EEG ROC-A1', 'EEG LOC-A2']

""" # Parkinson
datpath = 'RBDdataPark/dat/'
label0path = 'RBDdataPark/0/'
label1path = 'RBDdataPark/1/'
kept_channels = None
"""

os.makedirs(datpath, exist_ok=True)

datapath = label1path
for dirpath, dirnames, filenames in os.walk(datapath):
    for filename in filenames:
        if filename.endswith(".edf"):
            print_cyan("Processing {}".format(filename))
            raw = mne.io.read_raw_edf(os.path.join(dirpath, filename), verbose=0, preload=True)


            raw = preprocess_raw(raw, low_fq=.5, high_fq=20, kept_channels=kept_channels)
            raw.save(os.path.join(datpath, filename.split('.')[0] + '.fif'), overwrite=True)
