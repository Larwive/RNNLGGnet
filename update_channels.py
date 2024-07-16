import mne
import os
import warnings

warnings.filterwarnings("ignore")

# Remove
channels_to_remove = {
    #'Air cannula',
    #'Air cannual',
    #'Air Cannula',
    #'EOG Left',
    #'EOG Right'
    #'EEG T3-A2',
    #'EEG T4-A1'
    'EEG A1-A2'
}

for dirpath, dirnames, filenames in os.walk('./RBDdata/'):
    for filename in filenames:
        if filename.endswith(".fif"):
            raw = mne.io.read_raw_fif(os.path.join(dirpath, filename), verbose=False, preload=True)
            if len(channels_to_remove-set(raw.ch_names)) < len(channels_to_remove):
                subset = list(set(raw.ch_names) - channels_to_remove)
                print(raw.ch_names)
                print(subset)
                raw_subset = raw.copy().pick(subset)
                raw_subset.save(os.path.join(dirpath, filename), overwrite=True)


# Rename

renames = {
    #'Snore': 'Snoring Snore',
    #'EEG EKG-Ref': 'ECG EKG'
    #'Leg LEMG2'
}

for dirpath, dirnames, filenames in os.walk('./RBDdata/'):
    continue
    for filename in filenames:
        if filename.endswith(".fif"):
            raw = mne.io.read_raw_fif(os.path.join(dirpath, filename), verbose=False, preload=True)
            if len(renames.keys()-set(raw.ch_names)) < len(renames):
                print(raw.ch_names)
                raw_subset = raw.rename_channels(renames)
                raw_subset.save(os.path.join(dirpath, filename), overwrite=True)

"""
09318204 and 12984253: Snore (Snoring snore ?), Flow patient (Airflow ?), Effort tho (Resp Thorax ?), Effort abd (Resp Abdomen ?), SpO2 (SaO2 SpO2 ?), Leg LEMG1 (EMG Left_Leg ?), Leg LEMG2 (EMG Right_Leg ?)
No equivalent for: ECG I, RR, Body, Imp, EEG A1-A2 ?

"""
