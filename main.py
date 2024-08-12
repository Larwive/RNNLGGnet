import argparse
from cross_validation import *
from prepare_data import *


def banner_print(message) -> None:
    """
    Print a banner.
    :param message: Message to print
    :return: None
    """
    border = '*' * (len(message) + 4)
    print(f'{border}\n* {message} *\n{border}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data ########
    parser.add_argument('--dataset', type=str, default='HOSP')
    parser.add_argument('--label-type', type=str, default='park', choices=['rbd', 'park'])
    parser.add_argument('--data-path', type=str, default='./RBDdataPark/dat/')
    parser.add_argument('--subjects', type=int, default=19)
    parser.add_argument('--start-subject', type=int, default=0)
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
    parser.add_argument('--gamma', type=float, default=0.99)

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
    parser.add_argument('--graph-type', type=str, default='hem', choices=['fro', 'gen', 'hem', 'BL'])
    parser.add_argument('--hidden', type=int, default=32)

    # Reproduce the result using the saved model ######
    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()
    if args.label_type == 'rbd':
        args.data_path = './RBDdat/'
        args.subjects = 39
    assert args.start_phase <= args.end_phase

    sub_to_run = np.arange(args.start_subject, args.start_subject + args.subjects)
    pd = PrepareData(args)
    pd.run(sub_to_run, split=True, expand=True)
    cv = CrossValidation(args)
    seed_all(args.random_seed)

    if args.start_phase <= 1 <= args.end_phase:
        banner_print("Phase 1")
        cv.subject_fold_CV(subjects=sub_to_run, rand_state=args.kfold_rand_state)
    if args.start_phase <= 2 <= args.end_phase:
        banner_print("Phase 2")
        cv.subject_fold_cv_phase_2_3(subjects=sub_to_run, phase=2)
    if args.start_phase <= 3 <= args.end_phase:
        banner_print("Phase 3")
        cv.subject_fold_cv_phase_2_3(subjects=sub_to_run, phase=3)
