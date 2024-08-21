import matplotlib.pyplot as plt
import _pickle as cPickle

hdf_paths = [
    # Your paths here.
]
fig, ax = plt.subplots()

for path in hdf_paths:
    data = cPickle.load(open(path, 'rb'), encoding='latin1')['data']

    for i, ch_data in enumerate(data):
        ax.plot(ch_data + i * 10, label=f'Raw1 - Ch{i}')

plt.show()
