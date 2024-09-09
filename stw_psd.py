import mne
import matplotlib.pyplot as plt
import numpy as np


def compute_psd(path, compute_frequency, epoch_duration) -> None:
    """
    Compute the PSD epoch by epoch and plot it little by little.
    :param path: The path to the fif file.
    :param compute_frequency: The frequency at which compute the PSD epoch.
    :param epoch_duration: The duration of epoch to take for PSD.
    :return: None
    """
    plt.figure()
    raw = mne.io.read_raw_fif(path, preload=True)
    raw.pick_channels(['F4', 'F3', 'O1', 'O2', 'C3', 'C4'])
    raw = mne.set_bipolar_reference(raw, ['F3', 'F4', 'C3', 'C4'], ['C3', 'C4', 'O1', 'O2'], drop_refs=False)
    sfreq = int(raw.info['sfreq'])
    duration = int(raw.n_times / sfreq)
    X = [i/compute_frequency for i in range(int(compute_frequency*(duration+1)))]
    orig_ch_names = raw.info['ch_names']
    orig_sfreq = raw.info['sfreq']

    custom_labels = []
    custom_ticks = []
    for i in range(len(raw.info['ch_names'])):
        custom_labels.extend([raw.info['ch_names'][i], 'd{}/dt'.format(raw.info['ch_names'][i])])
        custom_ticks.extend([10*i, 10*i+5])

    info_cropped = mne.create_info(orig_ch_names, orig_sfreq, ch_types=['eeg'] * len(raw.info['ch_names']))
    info_cropped['description'] = "Cropped EEG data"
    points = int(epoch_duration * sfreq)
    psd_dict = {}
    der_dict = {}

    for i in range(len(raw.info['ch_names'])):
        psd_dict[str(i)] = []
        der_dict[str(i)] = [0]

    for i in range(duration * compute_frequency - points):
        cropped = raw.get_data()[:, int(i * sfreq / compute_frequency):int(i * sfreq / compute_frequency) + points]

        raw_cropped = mne.io.RawArray(cropped, info_cropped, verbose=0)
        raw_cropped.set_meas_date(raw.info['meas_date'])
        psd, frequencies = mne.time_frequency.psd_array_multitaper(raw_cropped.get_data(), sfreq=sfreq,
                                                                   fmin=2,
                                                                   fmax=3, verbose=0)
        arr = np.array(psd)
        arr = arr.mean(axis=1)
        for j in range(len(arr)):
            psd_dict[str(j)].append(arr[j])
            der_dict[str(j)].append((arr[j]-psd_dict[str(j)][-1-int(i>0)])/compute_frequency)
        if i % 500 == 0:
            plt.clf()
            plt.yticks([])
            for label, y in zip(custom_labels, custom_ticks):
                plt.text(-0.1, y, label, ha='center', va='center', fontsize=12, color='black',
                         transform=plt.gca().get_yaxis_transform())

            for j in range(len(arr)):
                plt.plot(X[:len(psd_dict[str(j)])], np.array(psd_dict[str(j)])+10*j)
                plt.plot(X[:len(psd_dict[str(j)])+1], np.array(der_dict[str(j)])+10*j+5)
            plt.pause(0.001)
    print("Done.")
    plt.xlabel("Time")
    plt.ylabel("PSD (μV²/Hz)")
    plt.title("PSD over time ({} s/epoch)".format(epoch_duration))
    plt.show()


if __name__ == '__main__':
    # compute_psd('stw/102-10-21.fif', 5, 3.7)
    compute_psd('stw/102-10-21.fif', 5, 3.7)
