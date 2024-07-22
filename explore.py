import mne
import os
import warnings

warnings.filterwarnings("ignore")
first = True
excluded, inter, excluded_events, inter_events = [], [], [], []
count = 0
chann_dict = {}
for dirpath, dirnames, filenames in os.walk('./RBDdata/'):
    for filename in filenames:
        if filename.endswith(".edf"):
            count += 1
            raw = mne.io.read_raw_edf(os.path.join(dirpath, filename), verbose=False)
            if first:
                inter = raw.ch_names
                inter_events = list(list(mne.events_from_annotations(raw, verbose=0))[1].keys())
                first = False
                for chan in inter:
                    chann_dict[chan] = 1


                if 'Snoring Snore' not in inter:
                    print(filename)
                    print(inter)
            else:
                for chan in raw.ch_names:
                    if chan in chann_dict.keys():
                        chann_dict[chan] += 1
                    else:
                        chann_dict[chan] = 1
                inter = [ch for ch in inter if ch in raw.ch_names]
                excluded = excluded+[ch for ch in raw.ch_names if ch not in inter and ch not in excluded]
                events = list(list(mne.events_from_annotations(raw, verbose=0))[1].keys())
                inter_events = [ev for ev in inter_events if ev in events]
                excluded_events = excluded_events+[ev for ev in events if ev not in inter_events and ev not in excluded_events]

                if 'Snoring Snore' not in raw.ch_names:
                    print(filename)
                    print(raw.ch_names)

print("Excluded channels:\n", excluded)
print("Intersection channels:\n", inter)
print("Excluded events:\n", [str(ev) for ev in excluded_events])
print("Intersection events:\n", [str(ev) for ev in inter_events])
print({k: v for k, v in sorted(chann_dict.items(), key=lambda item: item[1])})
print(count)


