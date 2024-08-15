import _pickle as cPickle
from argparse import Namespace
from train import *


class PrepareData:
    def __init__(self, args: Namespace):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path
        if args.label_type == 'park':
            self.original_order = ['EEG F3-A2', 'EEG F4-A1', 'EEG C3-A2', 'EEG C4-A1', 'EEG O1-A2', 'EEG O2-A1',
                                   'EEG LOC-A2', 'EEG ROC-A1', 'EMG Chin', 'ECG EKG', 'EMG Left_Leg', 'EMG Right_Leg',
                                   'Snoring Snore', 'Airflow', 'Resp Thorax', 'Resp Abdomen', 'Manual']
            self.graph_fro = [['EEG F3-A2'], ['EEG F4-A1'], ['EEG C3-A2', 'EEG C4-A1'], ['EEG O1-A2', 'EEG O2-A1'],
                              ['EEG LOC-A2', 'EEG ROC-A1'], ['EMG Left_Leg'], ['EMG Right_Leg'], ['EMG Chin'],
                              ['ECG EKG'],
                              ['Snoring Snore'], ['Airflow'], ['Resp Thorax'], ['Resp Abdomen'], ['Manual']]

            self.graph_gen = [['EEG F3-A2', 'EEG F4-A1'], ['EEG C3-A2', 'EEG C4-A1'], ['EEG O1-A2', 'EEG O2-A1'],
                              ['EEG LOC-A2', 'EEG ROC-A1'], ['EMG Left_Leg', 'EMG Right_Leg'], ['EMG Chin'],
                              ['ECG EKG'],
                              ['Snoring Snore'], ['Airflow'], ['Resp Thorax', 'Resp Abdomen'], ['Manual']]

            self.graph_hem = [['EEG F3-A2'], ['EEG F4-A1'], ['EEG C3-A2'], ['EEG C4-A1'], ['EEG O1-A2'], ['EEG O2-A1'],
                              ['EEG LOC-A2'], ['EEG ROC-A1'], ['EMG Left_Leg'], ['EMG Right_Leg'], ['EMG Chin'],
                              ['ECG EKG'],
                              ['Snoring Snore'], ['Airflow'], ['Resp Thorax'], ['Resp Abdomen'], ['Manual']]

            self.TS = ['EEG F3-A2', 'EEG F4-A1', 'EEG C3-A2', 'EEG C4-A1', 'EEG O1-A2', 'EEG O2-A1', 'EEG LOC-A2',
                       'EEG ROC-A1', 'EMG Chin', 'ECG EKG', 'EMG Left_Leg', 'EMG Right_Leg', 'Snoring Snore', 'Airflow',
                       'Resp Thorax', 'Resp Abdomen', 'Manual']
        else:  # 'rbd'
            self.original_order = ['EEG O1-A2', 'EEG O2-A1', 'ECG EKG']
            self.graph_fro = [['EEG O1-A2', 'EEG O2-A1'], ['ECG EKG']]
            self.graph_gen = [['EEG O1-A2', 'EEG O2-A1'], ['ECG EKG']]
            self.graph_hem = [['EEG O1-A2'], ['EEG O2-A1'], ['ECG EKG']]
            self.TS = ['EEG O1-A2', 'EEG O2-A1', 'ECG EKG']

        self.graph_type = args.graph_type

    def run(self, subject_list, split: bool = False, expand: bool = True, forced_graph=None,
            verbose: float = True) -> None:
        """
        The processed data will be saved './data_<dataset>/sub0.hdf'
        :param subject_list: the subjects that need to be processed
        :param split: whether to split one trial's data into shorter segment
        :param expand: whether to add an empty dimension for CNN
        :return: None
        """
        for sub in subject_list:
            data_, label_ = self.load_data_per_subject(sub, forced_graph, verbose=verbose)
            # select label type here
            label_ = self.label_selection(label_)

            if expand:
                # expand one dimension for deep learning(CNNs)
                data_ = np.expand_dims(data_, axis=-3)

            if split:
                data_, label_ = self.split(
                    data=data_, label=label_, segment_length=self.args.segment,
                    overlap=self.args.overlap, sampling_rate=self.args.sampling_rate, verbose=verbose)
            if verbose:
                print('Data and label prepared!\ndata:{} label:{}\n----------------------'.format(data_.shape,
                                                                                                  label_.shape))
            self.save(data_, label_, sub)

    def load_data_per_subject(self, sub: int, forced_graph=None, verbose:float = True) -> tuple[np.ndarray, np.ndarray]:
        """
        This function loads the target subject's original file
        :param sub: which subject to load
        :return: data: (??, ??, ????) label: (??, ?)
        """
        sub += 1
        sub_code = 's{}-'.format(sub)
        data_lis, label_lis = [], []
        for dirpath, dirnames, filenames in os.walk(self.data_path):
            for filename in filenames:
                if filename.startswith(sub_code):
                    subject_path = os.path.join(dirpath, filename)
                    subject = cPickle.load(open(subject_path, 'rb'), encoding='latin1')
                    label_lis.append(subject['labels'])
                    data_lis.append(subject['data'])
        # label = np.concatenate(label_lis, axis=0)
        label = np.array(label_lis)
        data = np.concatenate(data_lis, axis=1)
        #   data: ?? x ?? x ????
        #   label: ?? x ?
        # reorder the EEG channel to build the local-global graphs
        data = self.reorder_channel(data=data, graph=self.graph_type, forced_graph=forced_graph)
        if verbose:
            print('data:{} label:{}'.format(data.shape, label.shape))
        return data, label

    def reorder_channel(self, data: np.ndarray, graph: str, forced_graph=None) -> np.ndarray:
        """
        This function reorder the channel according to different graph designs
        :param data: (trial, channel, data)
        :param graph: graph type
        :return: reordered data: (trial, channel, data)
        """
        if forced_graph is not None:
            graph_idx = forced_graph
        elif graph == 'fro':
            graph_idx = self.graph_fro
        elif graph == 'gen':
            graph_idx = self.graph_gen
        elif graph == 'hem':
            graph_idx = self.graph_hem
        elif graph == 'BL':
            graph_idx = self.original_order
        elif graph == 'TS':
            graph_idx = self.TS
        else:
            graph_idx = self.original_order

        idx = []
        if graph in ['BL', 'TS']:
            for chan in graph_idx:
                idx.append(self.original_order.index(chan))
        else:
            num_chan_local_graph = []
            for i in range(len(graph_idx)):
                num_chan_local_graph.append(len(graph_idx[i]))
                for chan in graph_idx[i]:
                    idx.append(self.original_order.index(chan))

            # save the number of channels in local graph for building the LGG model in utils.py
            dataset = h5py.File('num_chan_local_graph_{}.hdf'.format(graph), 'w')
            dataset['data'] = num_chan_local_graph
            dataset.close()
        return data[idx, :]

    @staticmethod
    def label_selection(label):
        """
        Selects which dimension of labels to use and create binary label.
        :param label: (1)
        :return: (trial,)
        """
        return label[0]

    def save(self, data, label, sub: int) -> None:
        """
        This function save the processed data into target folder
        :param data: the processed data
        :param label: the corresponding label
        :param sub: the subject ID
        :return: None
        """
        save_path = os.getcwd()
        data_type = 'data_{}'.format(self.args.dataset)
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        name = 'sub{}.hdf'.format(sub)
        save_path = osp.join(save_path, name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()

    @staticmethod
    def split(data: np.ndarray, label, segment_length: float = 1, overlap: float = 0, sampling_rate: int = 256,
              verbose: float = True) -> tuple[np.ndarray, np.ndarray]:
        """
        This function split one trial's data into shorter segments
        :param data: (trial, f, channel, data)
        :param label: (trial,)
        :param segment_length: how long each segment is in seconds
        :param overlap: overlap rate
        :param sampling_rate: sampling rate
        :return: data: (trial, num_segment, f, channel, segment_length)
        label: (trial, num_segment,)
        """
        data_shape = data.shape
        step = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length
        data_split = []

        number_segment = int((data_shape[-1] - data_segment) // step)
        for i in range(number_segment + 1):
            data_split.append(data[:, :, (i * step):(i * step + data_segment)])
        data_split_array = np.stack(data_split, axis=1)
        label = np.stack([np.repeat(label, int(number_segment + 1))], axis=0)
        if verbose:
            print("The data and label are split: Data shape:{} Label:{}".format(data_split_array.shape, label.shape))
        data = data_split_array
        assert len(data) == len(label)
        return data, label
