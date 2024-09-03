import argparse
import mne
import torch
from torch.utils.data import DataLoader
import numpy as np

from utils import load_LGG, load_RNNLGG, load_resnet
from preprocess import preprocess_raw
from prepare_data import PrepareData

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

# Those with `# Warning` mean: Should be the same as used while training.
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
parser.add_argument('--lgg-path', type=str)  # To be filled when loading a RNNLGGnet model too.
parser.add_argument('--resnet-path', type=str)
parser.add_argument('--label-type', type=str, default='park', choices=['rbd', 'park'])  # Warning
parser.add_argument('--graph-type', type=str, default='hem', choices=['fro', 'gen', 'hem', 'BL'])
parser.add_argument('--input-shape', type=tuple, default=(1, 17, 512))  # Warning
parser.add_argument('--sampling-rate', type=int, default=128)  # Warning
parser.add_argument('--T', type=int, default=64)  # Warning
parser.add_argument('--hidden', type=int, default=32)  # Warning
parser.add_argument('--pool', type=int, default=16)  # Warning
parser.add_argument('--pool-step-rate', type=float, default=0.25)
parser.add_argument('--rnn-hidden-size', type=int, default=10)  # Warning
parser.add_argument('--rnn-num-layers', type=int, default=10)  # Warning
parser.add_argument('--segment-length', type=float, default=4.)  # Warning
parser.add_argument('--overlap', type=float, default=0.)
parser.add_argument('--batch-size', type=int, default=64)

parser.add_argument('--fif-path', type=str)  # Path to your fif file.
args = parser.parse_args()

# Load data
raw_data = mne.io.read_raw_fif(args.fif_path, preload=True)
preprocessed_data = preprocess_raw(raw_data)

# Prepare data
pd = PrepareData(args)
data = pd.prepare_raw_data(preprocessed_data, split=True, expand=True, segment_length=args.segment_length,
                           overlap=args.overlap)
data = np.expand_dims(np.concatenate(data, axis=0), axis=1)
data = torch.tensor(data, dtype=torch.float32, device=device)
dataloader = DataLoader(data, batch_size=args.batch_size)

## Load LGGNet

LGG_model = load_LGG(path=args.lgg_path, graph_type=args.graph_type, input_size=args.input_shape,
                     sampling_rate=args.sampling_rate, n_out_channels=args.T, n_hidden=args.hidden, pool=args.pool,
                     pool_step_rate=args.pool_step_rate)
LGG_model.eval()
LGG_model.to(device)
lgg_count0, lgg_count1 = 0, 0
for epoch in dataloader:
    lgg_pred = LGG_model(epoch)
    lgg_pred = (lgg_pred >= .5).int()
    lgg_count0 += len(epoch) - lgg_pred.sum()
    lgg_count1 += lgg_pred.sum()

## Load RNNLGGnet

RNNLGG_model = load_RNNLGG(path=args.path, LGG_path=args.lgg_path, graph_type=args.graph_type,
                           input_size=args.input_shape, sampling_rate=args.sampling_rate, n_out_channels=args.T,
                           n_hidden=args.hidden, pool=args.pool, pool_step_rate=args.pool_step_rate,
                           rnn_num_layers=args.rnn_num_layers, rnn_hidden_size=args.rnn_hidden_size)
RNNLGG_model.eval()
RNNLGG_model.to(device)
rnnlgg_count0, rnnlgg_count1 = 0, 0
for epoch in dataloader:
    rnnlgg_pred = RNNLGG_model(epoch)[0]
    rnnlgg_pred = (rnnlgg_pred >= .5).int()
    rnnlgg_count0 += len(epoch) - rnnlgg_pred.sum()
    rnnlgg_count1 += rnnlgg_pred.sum()

## Load ResNet

resnet = load_resnet(path=args.resnet_path, input_channels=args.input_shape[1])
resnet.eval()
resnet.to(device)
resnet_count0, resnet_count1 = 0, 0
for epoch in dataloader:
    resnet_pred = resnet(epoch)
    resnet_pred = (resnet_pred >= .5).int()
    resnet_count0 += len(epoch) - resnet_pred.sum()
    resnet_count1 += resnet_pred.sum()

# Prin results
print("LGG:\n'0': {}\n'1': {}".format(lgg_count0, lgg_count1))
print("RNNLGG:\n'0': {}\n'1': {}".format(rnnlgg_count0, rnnlgg_count1))
print("Resnet:\n'0': {}\n'1': {}".format(resnet_count0, resnet_count1))

"""
Example use:
python3 example.py --path 'save_overlap50_fro2/final_model_phase3.pth' --lgg-path 'save_overlap50_fro2/T_64_pool_16/model/sub0_phase1.pth' --resnet-path 'save_overlap50_park_resnet2/final_model.pth' --fif-path 'stw/102-10-21.fif' --batch-size 128 --overlap 0.5 --graph-type 'fro'
"""
