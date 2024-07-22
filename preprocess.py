import mne
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def update_channels(raw: mne.io.Raw) -> mne.io.Raw:
    channels_to_remove = {
        'Air cannula',
        'Air cannual',
        'Air Cannula',
        'EOG Left',
        'EOG Right'
        'EEG T3-A2',
        'EEG T4-A1'
        'EEG A1-A2'
    }

    renames = {
        'Snore': 'Snoring Snore',
        'EEG EKG-Ref': 'ECG EKG'
                       'Leg LEMG2'
    }

    if len(channels_to_remove - set(raw.ch_names)) < len(channels_to_remove):
        subset = list(set(raw.ch_names) - channels_to_remove)
        print(raw.ch_names)
        print(subset)
        raw = raw.pick(subset)

    if len(renames.keys() - set(raw.ch_names)) < len(renames):
        print(raw.ch_names)
        raw = raw.rename_channels(renames)
    return raw


def remove_muscle_artifacts(raw: mne.io.Raw, n_components: int = 15, method: str = "picard",
                            max_iter: str = "auto", random_state: int = 97, montage: str = "",
                            plot: bool = False) -> mne.io.Raw:
    """
    Remove muscle artifacts using ICA.
    :param raw: Raw data to remove artifacts from.
    :param n_components: See mne.preprocessing.ICA
    :param method: See mne.preprocessing.ICA
    :param max_iter: See mne.preprocessing.ICA
    :param random_state: See mne.preprocessing.ICA
    :param montage: The montage used when recording.
    :param plot: Whether to plot the ICA results.
    :return: Raw data with artifacts removed.
    """
    mne.datasets.eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage(montage)
    raw.set_montage(montage)
    raw.filter(l_freq=1.0, h_freq=None)

    ica = mne.preprocessing.ICA(
        n_components=n_components, method=method, max_iter=max_iter, random_state=random_state
    )
    ica.fit(raw)
    muscle_idx_auto, scores = ica.find_bads_muscle(raw)
    print(
        "Automatically found muscle artifact ICA components: "
        f"{muscle_idx_auto}")
    if plot:
        ica.plot_sources(raw)
        ica.plot_properties(raw, picks=muscle_idx_auto, log_scale=True)
        ica.plot_scores(scores, exclude=muscle_idx_auto)
    ica.exclude = muscle_idx_auto
    ica.apply(raw)
    return raw


def normalize(raw: mne.io.Raw) -> None:
    data, times = raw[:, :]
    data = data.T  # Shape should be (n_samples, n_channels)

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std

    raw._data = normalized_data.T


def preprocess_raw(raw: mne.io.Raw) -> mne.io.Raw:
    raw = update_channels(raw)
    raw.filter(l_freq=0.5, h_freq=50)
    raw = remove_muscle_artifacts(raw)

    raw.resample(128)
    normalize(raw)
    return raw


if __name__ == '__main__':
    for dirpath, dirnames, filenames in os.walk('./RBDdata/'):
        for filename in filenames:
            if filename.endswith(".fif"):
                pass
                # raw = mne.io.read_raw_fif(os.path.join(dirpath, filename), verbose=False, preload=True)
