import mne
import matplotlib.pyplot as plt
import numpy as np



def compute_psd(path, compute_frequency, epoch_duration):
    fig = plt.figure()
    raw = mne.io.read_raw_fif(path, preload=True)
    sfreq = int(raw.info['sfreq'])
    duration = int(raw.n_times / sfreq)
    orig_ch_names = raw.info['ch_names']
    orig_sfreq = raw.info['sfreq']

    info_cropped = mne.create_info(orig_ch_names, orig_sfreq, ch_types=['eeg'] * len(raw.info['ch_names']))
    info_cropped['description'] = "Cropped EEG data"
    points = int(epoch_duration*sfreq)
    psd_dict = {}
    plt.legend()

    for i in range(len(raw.info['ch_names'])):
        psd_dict[str(i)] = []

    for i in range(duration * compute_frequency - points):
        cropped = raw.get_data()[:, int(i * sfreq / compute_frequency):int(i * sfreq / compute_frequency)+points]

        raw_cropped = mne.io.RawArray(cropped, info_cropped, verbose=0)
        raw_cropped.set_meas_date(raw.info['meas_date'])
        psd, frequencies = mne.time_frequency.psd_array_multitaper(raw_cropped.get_data(), sfreq=sfreq,
                                                                               fmin=2,
                                                                               fmax=3, verbose=0)
        arr = np.array(psd)
        arr = arr.mean(axis=1)
        for j in range(len(arr)):
            psd_dict[str(j)].append(arr[j])
        if i % 500 == 0:
            plt.clf()
            for j in range(len(arr)):
                plt.plot(psd_dict[str(j)], label=raw.info['ch_names'][j])
            plt.legend()
            plt.pause(0.001)
    print("Done.")
    plt.xlabel("Time")
    plt.ylabel("PSD (μV²/Hz)")
    plt.title("PSD over time ({} s/epoch)".format(epoch_duration))
    plt.show()

if __name__ == '__main__':
    #compute_psd('stw/102-10-21.fif', 5, 3.7)
    compute_psd('stw/102-10-21.fif', 5, 3.7)
