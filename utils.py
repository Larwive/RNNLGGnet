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
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor

    def __init__(self, x_tensor, y_tensor):
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
    print('using gpu:', x)


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
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


def get_model(args):
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


def get_RNNLGG(args, excluded_subject: int, fold: int = 0, phase: int = 2):
    idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(args.graph_type), 'r')['data']))
    channels = sum(idx_local_graph)
    input_size = (args.input_shape[0], channels, args.input_shape[2])

    model_name_reproduce = 'sub{}_fold{}.pth'.format(excluded_subject, fold)
    data_type = 'model'
    experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
    load_path_final = osp.join(args.save_path, experiment_setting, data_type, model_name_reproduce)
    LGG = get_model(args)
    LGG.load_state_dict(torch.load(load_path_final, weights_only=False))
    model = RNNLGGNet(LGG, args.rnn_hidden_size, args.rnn_num_layers,
                      args.rnn_dropout, phase=phase, input_size=input_size,
                      sampling_rate=int(args.sampling_rate * args.scale_coefficient),
                      num_T=args.T, out_graph=args.hidden,
                      pool=args.pool, pool_step_rate=args.pool_step_rate,
                      idx_graph=idx_local_graph)
    return model


def get_dataloader(data, label, batch_size):
    dataset = eegDataset(data, label)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return loader


def get_metrics(y_pred, y_true, classes=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    if classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm


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
