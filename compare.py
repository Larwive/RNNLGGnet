from cross_validation import *
from prepare_data import *
import argparse
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def print_line():
    print('*' * os.get_terminal_size().columns)


parser = argparse.ArgumentParser()
# Data ########
parser.add_argument('--dataset', type=str, default='HOSP')
parser.add_argument('--segment', type=int, default=4)  # Length in second
parser.add_argument('--overlap', type=float, default=0)
parser.add_argument('--sampling-rate', type=int, default=128)
parser.add_argument('--scale-coefficient', type=float, default=1)
parser.add_argument('--input-shape', type=tuple, default=(1, 17, 512))
# Training Process ########
parser.add_argument('--random-seed', type=int, default=2021)
parser.add_argument('--max-epoch', type=int, default=200)
# Number of consecutive epochs without increase in accuracy of validation set before early stopping
parser.add_argument('--patient', type=int, default=5)
parser.add_argument('--max-epoch-cmb', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--step-size', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.2)

parser.add_argument('--save-path', default='./save/')
parser.add_argument('--load-path', default='./save/max-acc_phase{}.pth')
parser.add_argument('--load-path-final', default='./save/final_model_phase{}.pth')
parser.add_argument('--gpu', default='0')
parser.add_argument('--save-model', type=bool, default=True)
parser.add_argument('--kfold_rand_state', type=int, default=5)

parser.add_argument('--rnn-hidden-size', type=int, default=10)
parser.add_argument('--rnn-num-layers', type=int, default=10)
parser.add_argument('--rnn-dropout', type=float, default=.2)
parser.add_argument('--start-phase', type=int, default=1)
parser.add_argument('--end-phase', type=int, default=3)
parser.add_argument('--phase-2-epochs', type=int, default=200)
parser.add_argument('--phase-3-epochs', type=int, default=200)

# Model Parameters ########
parser.add_argument('--model', type=str, default='HOSPNet')
parser.add_argument('--pool', type=int, default=16)
parser.add_argument('--pool-step-rate', type=float, default=0.25)
parser.add_argument('--T', type=int, default=64)
parser.add_argument('--hidden', type=int, default=32)

parser.add_argument('--graph-type', type=str, default='hem', choices=['fro', 'gen', 'hem', 'BL'])
######## Reproduce the result using the saved model ######
args = parser.parse_args()
args.reproduce = True

sub_to_runs = [
    np.arange(0, 19),
    np.arange(0, 19),
    np.arange(0, 39),
    np.arange(0, 39),
]

train_labels = ["RBD-Parkinson", "RBD-Parkinson", "Healthy-RBD", "Healthy-RBD"]

xlabels = ["Subject number's models",
           "Subject number's models",
           "Subject number's models",
           "Subject number's models"
           ]

data_paths = ['./RBDdataPark/dat/', './RBDdataPark/dat/', './RBDdat/', './RBDdat/']
model_types = ['RNNLGGnet', 'resnet', 'RNNLGGnet', 'resnet']
model_pathss = [[
    './save_overlap50_fro/',
    './save_overlap50_fro2/',
    './save_overlap50_gen/',
    './save_overlap50_gen2/',
    './save_overlap50_hem/',
    './save_overlap50_hem2/',
    './save_overlap50_EEG_fro/',
    './save_overlap50_EEG_fro2/',
    './save_overlap50_EEG_gen/',
    './save_overlap50_EEG_gen2/',
    './save_overlap50_EEG_hem/',
    './save_overlap50_EEG_hem2/',
    './save_overlap50_EMG_fro/',
    './save_overlap50_EMG_fro2/',
    './save_overlap50_EMG_gen/',
    './save_overlap50_EMG_gen2/',
    './save_overlap50_oth_fro/',
    './save_overlap50_oth_fro2/',
    './save_overlap50_oth_gen/',
    './save_overlap50_oth_gen2/',
], [
    './save_overlap50_park_resnet/',
], [
    './save_overlap50_rbd_fro/',
    './save_overlap50_rbd_fro2/',
    './save_overlap50_rbd_hem/',
    './save_overlap50_rbd_hem2/',
], [
    './save_overlap50_rbd_resnet/',
    './save_overlap50_rbd_resnet2/',
]]

input_shapess = [[
    (1, 17, 512),
    (1, 17, 512),
    (1, 17, 512),
    (1, 17, 512),
    (1, 17, 512),
    (1, 17, 512),
    (1, 8, 512),
    (1, 8, 512),
    (1, 8, 512),
    (1, 8, 512),
    (1, 8, 512),
    (1, 8, 512),
    (1, 3, 512),
    (1, 3, 512),
    (1, 3, 512),
    (1, 3, 512),
    (1, 6, 512),
    (1, 6, 512),
    (1, 6, 512),
    (1, 6, 512),
], [
    (1, 17, 512),
], [
    (1, 3, 512),
    (1, 3, 512),
    (1, 3, 512),
    (1, 3, 512),
], [
    (1, 3, 512),
    (1, 3, 512)
]]

labelss = [[
    "All channels, fro graph",
    "All channels, fro graph 2",
    "All channels, gen graph",
    "All channels, gen graph 2",
    "All channels, hem graph",
    "All channels, hem graph 2",
    "EEG channels, fro graph",
    "EEG channels, fro graph 2",
    "EEG channels, gen graph",
    "EEG channels, gen graph2",
    "EEG channels, hem graph",
    "EEG channels, hem graph2",
    "EMG channels, fro graph",
    "EMG channels, fro graph2",
    "EMG channels, gen graph",
    "EMG channels, gen graph2",
    "Oth channels, fro graph",
    "Oth channels, fro graph2",
    "Oth channels, gen graph",
    "Oth channels, gen graph2",
], [
    "All channels",
], [
    "All channels, fro graph",
    "All channels, fro graph2",
    "All channels, hem graph",
    "All channels, hem graph2",
], [
    "All channels",
    "All channels2",
]]

label_types = ["park", "park", "rbd", "rbd"]

channels_lists = [[
    [['EEG F3-A2'], ['EEG F4-A1'], ['EEG C3-A2', 'EEG C4-A1'], ['EEG O1-A2', 'EEG O2-A1'],
     ['EEG LOC-A2', 'EEG ROC-A1'], ['EMG Left_Leg'], ['EMG Right_Leg'], ['EMG Chin'],
     ['ECG EKG'], ['Snoring Snore'], ['Airflow'], ['Resp Thorax'], ['Resp Abdomen'], ['Manual']],
    [['EEG F3-A2'], ['EEG F4-A1'], ['EEG C3-A2', 'EEG C4-A1'], ['EEG O1-A2', 'EEG O2-A1'],
     ['EEG LOC-A2', 'EEG ROC-A1'], ['EMG Left_Leg'], ['EMG Right_Leg'], ['EMG Chin'],
     ['ECG EKG'], ['Snoring Snore'], ['Airflow'], ['Resp Thorax'], ['Resp Abdomen'], ['Manual']],
    [['EEG F3-A2', 'EEG F4-A1'], ['EEG C3-A2', 'EEG C4-A1'], ['EEG O1-A2', 'EEG O2-A1'],
     ['EEG LOC-A2', 'EEG ROC-A1'], ['EMG Left_Leg', 'EMG Right_Leg'], ['EMG Chin'],
     ['ECG EKG'], ['Snoring Snore'], ['Airflow'], ['Resp Thorax', 'Resp Abdomen'], ['Manual']],
    [['EEG F3-A2', 'EEG F4-A1'], ['EEG C3-A2', 'EEG C4-A1'], ['EEG O1-A2', 'EEG O2-A1'],
     ['EEG LOC-A2', 'EEG ROC-A1'], ['EMG Left_Leg', 'EMG Right_Leg'], ['EMG Chin'],
     ['ECG EKG'], ['Snoring Snore'], ['Airflow'], ['Resp Thorax', 'Resp Abdomen'], ['Manual']],
    [['EEG F3-A2'], ['EEG F4-A1'], ['EEG C3-A2'], ['EEG C4-A1'], ['EEG O1-A2'], ['EEG O2-A1'],
     ['EEG LOC-A2'], ['EEG ROC-A1'], ['EMG Left_Leg'], ['EMG Right_Leg'], ['EMG Chin'],
     ['ECG EKG'], ['Snoring Snore'], ['Airflow'], ['Resp Thorax'], ['Resp Abdomen'], ['Manual']],
    [['EEG F3-A2'], ['EEG F4-A1'], ['EEG C3-A2'], ['EEG C4-A1'], ['EEG O1-A2'], ['EEG O2-A1'],
     ['EEG LOC-A2'], ['EEG ROC-A1'], ['EMG Left_Leg'], ['EMG Right_Leg'], ['EMG Chin'],
     ['ECG EKG'], ['Snoring Snore'], ['Airflow'], ['Resp Thorax'], ['Resp Abdomen'], ['Manual']],

    [['EEG F3-A2'], ['EEG F4-A1'], ['EEG C3-A2', 'EEG C4-A1'], ['EEG O1-A2', 'EEG O2-A1'],
     ['EEG LOC-A2', 'EEG ROC-A1']],
    [['EEG F3-A2'], ['EEG F4-A1'], ['EEG C3-A2', 'EEG C4-A1'], ['EEG O1-A2', 'EEG O2-A1'],
     ['EEG LOC-A2', 'EEG ROC-A1']],
    [['EEG F3-A2', 'EEG F4-A1'], ['EEG C3-A2', 'EEG C4-A1'], ['EEG O1-A2', 'EEG O2-A1'],
     ['EEG LOC-A2', 'EEG ROC-A1']],
    [['EEG F3-A2', 'EEG F4-A1'], ['EEG C3-A2', 'EEG C4-A1'], ['EEG O1-A2', 'EEG O2-A1'],
     ['EEG LOC-A2', 'EEG ROC-A1']],
    [['EEG F3-A2'], ['EEG F4-A1'], ['EEG C3-A2'], ['EEG C4-A1'], ['EEG O1-A2'], ['EEG O2-A1'],
     ['EEG LOC-A2'], ['EEG ROC-A1']],
    [['EEG F3-A2'], ['EEG F4-A1'], ['EEG C3-A2'], ['EEG C4-A1'], ['EEG O1-A2'], ['EEG O2-A1'],
     ['EEG LOC-A2'], ['EEG ROC-A1']],

    [['EMG Left_Leg'], ['EMG Right_Leg'], ['EMG Chin']],
    [['EMG Left_Leg'], ['EMG Right_Leg'], ['EMG Chin']],
    [['EMG Left_Leg', 'EMG Right_Leg'], ['EMG Chin']],
    [['EMG Left_Leg', 'EMG Right_Leg'], ['EMG Chin']],

    [['ECG EKG'], ['Snoring Snore'], ['Airflow'], ['Resp Thorax'], ['Resp Abdomen'], ['Manual']],
    [['ECG EKG'], ['Snoring Snore'], ['Airflow'], ['Resp Thorax'], ['Resp Abdomen'], ['Manual']],
    [['ECG EKG'], ['Snoring Snore'], ['Airflow'], ['Resp Thorax', 'Resp Abdomen'], ['Manual']],
    [['ECG EKG'], ['Snoring Snore'], ['Airflow'], ['Resp Thorax', 'Resp Abdomen'], ['Manual']],
], [
    [['EEG F3-A2', 'EEG F4-A1', 'EEG C3-A2', 'EEG C4-A1', 'EEG O1-A2', 'EEG O2-A1',
      'EEG LOC-A2', 'EEG ROC-A1', 'EMG Chin', 'ECG EKG', 'EMG Left_Leg', 'EMG Right_Leg',
      'Snoring Snore', 'Airflow', 'Resp Thorax', 'Resp Abdomen', 'Manual']]
], [
    [['EEG O1-A2', 'EEG O2-A1'], ['ECG EKG']],
    [['EEG O1-A2', 'EEG O2-A1'], ['ECG EKG']],
    [['EEG O1-A2'], ['EEG O2-A1'], ['ECG EKG']],
    [['EEG O1-A2'], ['EEG O2-A1'], ['ECG EKG']]
], [
    [['EEG O1-A2', 'EEG O2-A1', 'ECG EKG']],
    [['EEG O1-A2', 'EEG O2-A1', 'ECG EKG']]
]]

fign = 0

plots = [0, 1, 2, 3]
plotting = -1

data_test, label_test = None, None
for model_paths, input_shapes, labels, channels_lists, train_label, sub_to_run, xlabel, data_path, label_type, model_type in zip(
        model_pathss,
        input_shapess,
        labelss,
        channels_lists,
        train_labels,
        sub_to_runs,
        xlabels,
        data_paths, label_types, model_types):
    args.data_path = data_path
    args.subjects = sub_to_run[-1] + 1
    args.label_type = label_type
    args.model_type = model_type
    plotting += 1
    if plotting not in plots:
        continue
    accuracies, stds = [], []
    for model_path, input_shape, label, channels in zip(model_paths, input_shapes, labels, channels_lists):
        if model_type == 'RNNLGGnet':
            max_phase = 3
        else:  # resnet
            max_phase = 1
        args.save_path = model_path
        args.input_shape = input_shape
        pd = PrepareData(args)
        pd.run(sub_to_run, split=True, expand=True, forced_graph=channels, verbose=False)
        for phase in range(1, max_phase + 1):
            print_red("{} phase {}".format(model_path, phase))
            accuracies_sub = []

            cv = CrossValidation(args)
            seed_all(args.random_seed)

            acc, std, accs, data_test, label_test = cv.compare(subjects=sub_to_run,
                                                               data_test=data_test,
                                                               label_test=label_test,
                                                               phase=phase)
            plt.figure(fign)
            plt.bar(np.arange(len(accs)), accs, label=label)
            for i in sub_to_run:
                accuracies_sub.append(np.array(accs[i * 10:(i + 1) * 10]).mean())
            plt.figure(fign + 1)
            plt.bar(np.arange(len(accuracies_sub)), accuracies_sub, label=label)
            accuracies.append(acc)
            stds.append(std)
            data_test, label_test = None, None

    plt.figure(fign)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.xlabel('Test number')
    plt.ylabel('Accuracy')
    plt.title(train_label)

    plt.figure(fign + 1)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.xlabel('Subject number\'s models')
    plt.ylabel('Accuracy')
    plt.title(train_label)

    print("{}\nmodel: acc/std".format(train_label))
    print_line()
    for model, acc, std in zip(model_paths, accuracies, stds):
        print("{}: {}/{}".format(model, acc, std))
    print_line()
    fign += 2

plt.show()
