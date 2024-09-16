from sys import argv
import mne
from mne.time_frequency import tfr_multitaper
import matplotlib.pyplot as plt
import numpy as np

path = argv[1]

raw = mne.io.read_raw_fif(path, preload=True)
raw.pick_channels(['F4', 'F3', 'O1', 'O2', 'C3', 'C4', 'EEG ROC-A1', 'EEG LOC-A2'])
raw = mne.set_bipolar_reference(raw, ['F3', 'F4', 'C3', 'C4'], ['C3', 'C4', 'O1', 'O2'], drop_refs=False)

freqs = np.linspace(2, 4, 30)
n_cycles = 2/freqs

power = tfr_multitaper(raw, freqs=freqs, n_cycles=n_cycles, time_bandwidth=2.0, return_itc=False)
freq_band_idx = np.where((power.freqs >= 2) & (power.freqs <= 4))[0]
print(power.freqs)
power_data = power.data #[freq_band_idx]

threshold = 2.71**5

above_threshold = np.max(power_data, axis=0) > threshold

filtered_power = power_data[:, above_threshold]


#filtered_power.plot([0], baseline=(-0.5, 0), mode='logratio', title='Power Spectrum')
plt.imshow(np.log(filtered_power), aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Filtered Power Spectrum (Above Threshold) ({})'.format(path))
raw.plot()
plt.show()