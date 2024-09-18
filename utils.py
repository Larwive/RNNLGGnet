import os
import time
import h5py
import numpy as np
import pprint
import random
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import os.path as osp
from model import *


class eegDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        """
        Wrapper class for tensors.
        :param x_tensor: (sample, channel, datapoint(feature))
        :param y_tensor: (sample,) type = torch.tensor
        """
        if type(x_tensor) is np.ndarray:
            self.x = torch.tensor(x_tensor, dtype=torch.float32)
            shapex0 = self.x.shape[0]
        else:
            self.x = x_tensor.to(dtype=torch.float32)
            shapex0 = self.x.size(0)
        if type(y_tensor) is np.ndarray:
            self.y = torch.tensor(y_tensor, dtype=torch.float32)
            shapey0 = self.y.shape[0]
        else:
            self.y = y_tensor.to(dtype=torch.float32)
            shapey0 = self.y.size(0)
        assert shapex0 == shapey0

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


def set_gpu(x):
    torch.set_num_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    # print('using gpu:', x)


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Timer:
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = int((time.time() - self.o) / p)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def get_model(args) -> nn.Module:
    if args.model_type == 'resnet':
        return get_resnet(args)
    elif args.model_type == 'RNNLGGnet':
        return get_LGG(args)


def get_LGG(args) -> LGGNet:
    """
    Returns a blank LGGNet model.
    :param args: The arguments given to the main.
    :return: A blank LGGNet model.
    """
    idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(args.graph_type), 'r')['data']))
    channels = sum(idx_local_graph)
    input_size = (args.input_shape[0], channels, args.input_shape[2])
    model = LGGNet(input_size=input_size,
                   sampling_rate=int(args.sampling_rate * args.scale_coefficient),
                   num_T=args.T, out_graph=args.hidden,
                   dropout_rate=args.dropout,
                   pool=args.pool, pool_step_rate=args.pool_step_rate,
                   idx_graph=idx_local_graph)
    return model


def get_RNNLGG(args, excluded_subject: int, fold: int = 0, phase: int = 2) -> RNNLGGNet:
    """
    Returns a blank or pretrained RNNLGGNet model. To be used while training or reproducing results.
    :param args: The arguments given to the main.
    :param excluded_subject: The first excluded subject in the subject-wise cross-validation.
    :param fold: The current fold.
    :param phase: The current phase.
    :return: A blank or pretrained RNNLGGNet model.
    """
    idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(args.graph_type), 'r')['data']))
    channels = sum(idx_local_graph)
    input_size = (args.input_shape[0], channels, args.input_shape[2])

    model_name_reproduce = 'sub{}_phase{}.pth'.format(excluded_subject, phase - 1)
    data_type = 'model'
    experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
    load_path_final = osp.join(args.save_path, experiment_setting, data_type, model_name_reproduce)

    if phase == 2:
        previous_model = get_LGG(args)
        previous_model.load_state_dict(torch.load(load_path_final, weights_only=False))
    else:  # phase=3 here
        previous_model = get_RNNLGG(args, excluded_subject, fold, phase - 1)
    model = RNNLGGNet(LGG_model=previous_model, hidden_size=args.rnn_hidden_size, num_layers=args.rnn_num_layers,
                      dropout_rate=args.rnn_dropout, phase=phase, input_size=input_size,
                      sampling_rate=int(args.sampling_rate * args.scale_coefficient),
                      num_T=args.T, out_graph=args.hidden,
                      pool=args.pool, pool_step_rate=args.pool_step_rate,
                      idx_graph=idx_local_graph)
    if args.reproduce and phase >= 2:
        model_name_reproduce = 'sub{}_phase{}.pth'.format(excluded_subject, phase)
        load_path_final = osp.join(args.save_path, experiment_setting, data_type, model_name_reproduce)
        model.load_state_dict(torch.load(load_path_final, weights_only=False))
    return model


def get_resnet(args) -> ResNet:
    """
    Returns a blank ResNet model.
    :param args: The arguments given to the main.
    :return: A blank ResNet model.
    """
    model = ResNet(input_channels=args.input_shape[1])
    return model


def load_LGG(path, graph_type, input_size, sampling_rate, n_out_channels, n_hidden, pool, pool_step_rate):
    idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(graph_type), 'r')['data']))
    input_size = [input_size[0], sum(idx_local_graph), input_size[2]]
    model = LGGNet(input_size=input_size,
                   sampling_rate=int(sampling_rate),
                   num_T=n_out_channels, out_graph=n_hidden,
                   pool=pool, pool_step_rate=pool_step_rate,
                   idx_graph=idx_local_graph)
    model.load_state_dict(torch.load(path, weights_only=False))
    return model


def load_RNNLGG(path, LGG_path, graph_type, input_size, sampling_rate, n_out_channels, n_hidden, pool, pool_step_rate,
                rnn_hidden_size, rnn_num_layers):
    idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(graph_type), 'r')['data']))
    input_size = [input_size[0], sum(idx_local_graph), input_size[2]]

    LGG_model = load_LGG(LGG_path, graph_type=graph_type, input_size=input_size, sampling_rate=sampling_rate,
                         n_out_channels=n_out_channels, n_hidden=n_hidden, pool=pool, pool_step_rate=pool_step_rate)

    model_phase2 = RNNLGGNet(LGG_model=LGG_model, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
                      input_size=input_size,
                      sampling_rate=int(sampling_rate),
                      num_T=n_out_channels, out_graph=n_hidden,
                      pool=pool, pool_step_rate=pool_step_rate,
                      idx_graph=idx_local_graph)
    model = RNNLGGNet(LGG_model=model_phase2, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
                      input_size=input_size,
                      sampling_rate=int(sampling_rate),
                      num_T=n_out_channels, out_graph=n_hidden,
                      pool=pool, pool_step_rate=pool_step_rate,
                      idx_graph=idx_local_graph, phase=3)
    model.load_state_dict(torch.load(path, weights_only=False))
    return model


def load_resnet(path, input_channels):
    model = ResNet(input_channels=input_channels)
    model.load_state_dict(torch.load(path, weights_only=False))
    return model


def get_dataloader(data, label, batch_size):
    dataset = eegDataset(data, label)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return loader


def get_metrics(y_pred, y_true, get_cm: bool = True, classes=None):
    # if classes is None:
    #    classes = [0., 1.]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    if get_cm:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        return acc, f1, cm
    return acc, f1


def get_trainable_parameter_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def L1Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err


def L2Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(w.pow(2))
    return err


def print_cyan(string):
    print("\033[0;36m{}\033[0m".format(string))


def print_purple(string):
    print("\033[0;35m{}\033[0m".format(string))


def print_red(string):
    print("\033[0;31m{}\033[0m".format(string))
