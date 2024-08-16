import copy
from sklearn.model_selection import KFold
from functools import reduce
from train import *
from utils import Averager, ensure_path

ROOT = os.getcwd()


def subject_fold(subjects, rate: float):
    """
    Give groups of subjects for multiple-subject-wise cross-validation.
    :param subjects: All the subjects to train on.
    :param rate: Rate at which to split the subjects.
    :return: Yields sub lists of subjects to do cross-validation on.
    """
    group_size = int(len(subjects) * rate) + 1
    current = group_size
    while current < len(subjects) + group_size:
        yield subjects[max(0, current - group_size):min(len(subjects), current)]
        current += group_size


class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.subjects = args.subjects
        # Log the results per subject
        # result_path = osp.join(args.save_path, 'result')
        # ensure_path(result_path)
        # self.text_file = osp.join(result_path,
        #                          "results_{}.txt".format(args.dataset))
        """with open(self.text_file, 'a') as file:
            file.write("\n" + str(datetime.datetime.now()) +
                       "\nTrain:Parameter setting for " + str(args.model) + ' on ' + str(args.dataset) +
                       "\n2)random_seed:" + str(args.random_seed) +
                       "\n3)learning_rate:" + str(args.learning_rate) +
                       "\n4)pool:" + str(args.pool) +
                       "\n5)num_epochs:" + str(args.max_epoch) +
                       "\n6)batch_size:" + str(args.batch_size) +
                       "\n7)dropout:" + str(args.dropout) +
                       "\n8)hidden_node:" + str(args.hidden) +
                       "\n9)input_shape:" + str(args.input_shape) +
                       "\n11)T:" + str(args.T) +
                       "\n12)graph-type:" + str(args.graph_type) + '\n')"""

    @staticmethod
    def read_data(sub: int, save_path: str, data_type: str):
        """
        Read subject data from `hdf` files.
        :param sub: The subject number to read data of.
        :param save_path: The path to the `hdf` files.
        :param data_type: Subdirectory of `save_path`.
        :return: Data and label from `hdf` files of specified subject.
        """
        sub_code = 'sub{}.hdf'.format(sub)
        path = osp.join(save_path, data_type, sub_code)
        with h5py.File(path, 'r') as dataset:
            data = np.array(dataset['data'])
            label = np.array(dataset['label'])
        return data, label

    @staticmethod
    def augment_data(datas, labels, n_0, n_1):
        to_augment = 0 if n_0 < n_1 else 1
        diff = n_1 - n_0 if to_augment == 0 else n_0 - n_1
        factor = max(n_0 // n_1, n_1 // n_0)
        for i, (data, label) in enumerate(zip(datas, labels)):

            if int(label[0][0]) == to_augment:
                n_augmented = min(diff, data.shape[1])
                subsubdatas = []
                for _ in range(factor):
                    offset = np.random.randint(data.shape[1] - n_augmented + 1)
                    subsubdatas.append(data[:, offset:offset + n_augmented, :, :])
                    labels[i] = np.concatenate((labels[i], labels[i][:, offset:offset + n_augmented]), axis=1)

                subdata = np.concatenate(subsubdatas, axis=1)
                datas[i] = np.concatenate((data, subdata + np.random.randn(*subdata.shape)), axis=1)

                diff -= n_augmented

    @staticmethod
    def augment_data_global(datas, labels, num_0, num_1, max_0, max_1, rate: float = 1.):
        num_0 = int(num_0 * (1 - rate)) + (1 if num_0 > num_1 else 0)
        num_1 = int(num_1 * (1 - rate)) + (1 if num_1 > num_0 else 0)
        to_augment = 0 if num_1 * max_1 > num_0 * max_0 else 1
        target_max_to_augment = int(num_1 * max_1 / num_0) if to_augment == 0 else int(num_0 * max_0 / num_1)
        target_other = max_0 if to_augment == 1 else max_1
        counts = [0, 0]
        for i, (data, label) in enumerate(zip(datas, labels)):
            if int(label[0][0]) == to_augment:
                n_augmented = target_max_to_augment - data.shape[1]
            else:
                n_augmented = target_other - data.shape[1]
            if n_augmented < 1:
                continue
            act_augmented = 0
            while act_augmented < n_augmented:
                tmp_augment = min(n_augmented - act_augmented, data.shape[1])
                offset = np.random.randint(data.shape[1] - tmp_augment + 1)
                subdata = data[:, offset:offset + tmp_augment, :, :]
                datas[i] = np.concatenate((datas[i], subdata + np.random.randn(*subdata.shape)), axis=1)
                labels[i] = np.concatenate((labels[i], labels[i][:, offset:offset + tmp_augment]), axis=1)
                act_augmented += tmp_augment
            counts[int(label[0][0])] += datas[i].shape[1]

    def load_per_subject(self, sub: int, verbose: bool = True):
        """
        Load data for a subject.
        :param sub: The subject number to read data of.
        :param verbose: Whether to print additional information.
        :return: Data and label of the subject.
        """
        save_path = os.getcwd()
        data_type = 'data_{}'.format(self.args.dataset)
        data, label = self.read_data(sub, save_path, data_type)
        if verbose:
            print('>>> Data:{} Label:{}'.format(data.shape, label.shape))
        return data, label

    @staticmethod
    def reduce_data(datas, labels, shuffle: bool):
        datas = reduce(lambda x, y: np.concatenate((x, y), axis=1), datas)
        labels = reduce(lambda x, y: np.concatenate((x, y), axis=1), labels)
        if shuffle:
            permutation = np.random.permutation(len(datas))
            datas = datas[permutation]
            labels = labels[permutation]
        return datas, labels

    def load_all_except_some(self, excluded_subs, shuffle: bool = False, verbose: bool = True):
        """
        Load data for all subjects except one.
        :param excluded_subs: The subject numbers to exclude from loading.
        :param shuffle: Whether to shuffle the data.
        :param verbose: Whether to print additional information.
        :return: Datas and labels of wanted subjects.
        """
        save_path = os.getcwd()
        data_type = 'data_{}'.format(self.args.dataset)
        datas, labels = [], []
        len_0, len_1 = 0, 0
        for sub in range(self.subjects):
            if sub in excluded_subs:
                continue
            data, label = self.read_data(sub, save_path, data_type)
            datas.append(data)
            labels.append(label)
            if int(label[0][0]) == 0:
                len_0 += data.shape[1]
            else:
                len_1 += data.shape[1]
            if verbose:
                print('>>> Data:{} Label:{}'.format(datas[-1].shape, labels[-1].shape))
        self.augment_data(datas, labels, len_0, len_1)
        return self.reduce_data(datas, labels, shuffle)

    def load_subjects(self, subjects, shuffle: bool = False, verbose: bool = True):
        """
        Load some subjects' data.
        :param subjects: The subjects to load data of.
        :param shuffle: Whether to shuffle the data.
        :param verbose: Whether to print additional information.
        :return: Datas and labels of wanted subjects.
        """
        save_path = os.getcwd()
        data_type = 'data_{}'.format(self.args.dataset)
        datas, labels = [], []
        len_0, len_1 = 0, 0

        for sub in subjects:
            data, label = self.read_data(sub, save_path, data_type)
            datas.append(data)
            labels.append(label)
            if int(label[0][0]) == 0:
                len_0 += data.shape[1]
            else:
                len_1 += data.shape[1]
            if verbose:
                print('>>> Data:{} Label:{}'.format(datas[-1].shape, labels[-1].shape))
        self.augment_data(datas, labels, len_0, len_1)

        return self.reduce_data(datas, labels, shuffle)

    def load_all(self, prepare_data=False, expand=False, verbose: bool = True, rate: float = 1.):
        """
        Load all subjects' data.
        :param prepare_data: Whether to prepare the data.
        :param expand: Whether to expand data.
        :param verbose: Whether to print additional information.
        :param rate: See 'CrossValidation.augment_data_global'
        :return: Datas and labels of all subjects.
        """
        save_path = os.getcwd()
        data_type = 'data_{}'.format(self.args.dataset)
        datas, labels = [], []
        num_0, num_1 = 0, 0
        max_0, max_1 = 0, 0
        for sub in range(self.subjects):
            data, label = self.read_data(sub, save_path, data_type)
            if prepare_data:
                data, label = self.prepare_data_subject_fold(data, label)
            if expand:
                data = np.expand_dims(data, axis=1)
            datas.append(data)
            labels.append(label)
            if int(label[0][0]) == 0:
                num_0 += 1
                if data.shape[1] > max_0:
                    max_0 = data.shape[1]
            else:
                num_1 += 1
                if data.shape[1] > max_1:
                    max_1 = data.shape[1]
            if verbose:
                print('>>> Data:{} Label:{}'.format(datas[-1].shape, labels[-1].shape))
        self.augment_data_global(datas, labels, num_0, num_1, max_0, max_1, rate=rate)

        return datas, labels

    def prepare_data(self, idx_train, idx_test, data, label):
        """
        1. get training and testing data according to the index
        2. numpy.array-->torch.tensor
        :param idx_train: index of training data
        :param idx_test: index of testing data
        :param data: (segments, 1, channel, data)
        :param label: (segments,)
        :return: data and label
        """

        """
        We want to do trial-wise 10-fold, so the idx_train/idx_test is for
        trials.
        data: (trial, segment, 1, chan, datapoint)
        To use the normalization function, we should change the dimension from
        (trial, segment, 1, chan, datapoint) to (trial*segments, 1, chan, datapoint)
        """
        data_train = np.concatenate(data[idx_train], axis=0)
        label_train = np.concatenate(label[idx_train], axis=0)
        data_test = data[idx_test]
        label_test = label[idx_test]
        return self.normalize_data(data_train, label_train), self.normalize_data(data_test, label_test)

    def prepare_data_subject_fold(self, data, label):
        """
        Prepare data for the subject-wise cross-validation.
        :param data: Data to prepare.
        :param label: Label to prepare.
        :return: Prepared data and label.
        """

        """
        We want to do subject-wise fold, so the idx_train/idx_test is for
        trials.
        data: (trial, segment, 1, chan, datapoint)
        To use the normalization function, we should change the dimension from
        (trial, segment, 1, chan, datapoint) to (trial*segments, 1, chan, datapoint)
        """
        data = np.concatenate(data, axis=0)
        label = np.concatenate(label, axis=0)
        return self.normalize_data(data, label)

    @staticmethod
    def normalize_data(data, label):
        """
        Prepare the data format for training the model using PyTorch
        :param data: Data to be normalized
        :param label: Associated label
        :return: Normalized data and label
        """
        # Already normalized in the `.dat` files
        # data_train, data_test = self.normalize(train_data=data_train, test_data=data_test)
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).long()
        return data, label

    @staticmethod
    def normalize(train_data, test_data):
        """
        This function do standard normalization for EEG channel by channel
        :param train_data: training data (sample, 1, chan, datapoint)
        :param test_data: testing data (sample, 1, chan, datapoint)
        :return: normalized training and testing data
        """
        # data: sample x 1 x channel x data
        for channel in range(train_data.shape[2]):
            mean = np.mean(train_data[:, :, channel, :])
            std = np.std(train_data[:, :, channel, :])
            train_data[:, :, channel, :] = (train_data[:, :, channel, :] - mean) / std
            test_data[:, :, channel, :] = (test_data[:, :, channel, :] - mean) / std
        return train_data, test_data

    @staticmethod
    def split_balance_class(data, label, train_rate, randomize):
        """
        Get the validation set using the same percentage of the two classe samples
        :param data: training data (segment, 1, channel, data)
        :param label: (segments,)
        :param train_rate: the percentage of training data
        :param randomize: bool, whether to shuffle the training data before get the validation data
        :return: data_train, label_train, and data_val, label_val
        """
        np.random.seed(0)

        index_0 = np.where(label == 0)[0]
        index_1 = np.where(label == 1)[0]

        # for class 0
        index_random_0 = copy.deepcopy(index_0)

        # for class 1
        index_random_1 = copy.deepcopy(index_1)

        if randomize:
            np.random.shuffle(index_random_0)
            np.random.shuffle(index_random_1)

        index_train = np.concatenate((index_random_0[:int(len(index_random_0) * train_rate)],
                                      index_random_1[:int(len(index_random_1) * train_rate)]),
                                     axis=0)
        index_val = np.concatenate((index_random_0[int(len(index_random_0) * train_rate):],
                                    index_random_1[int(len(index_random_1) * train_rate):]),
                                   axis=0)

        # get validation
        val = data[index_val]
        val_label = label[index_val]

        train_data = data[index_train]
        train_label = label[index_train]

        return train_data, train_label, val, val_label

    def subject_fold_CV(self, subjects=None, shuffle=False, rand_state=None, rate: float = .2):
        """
        This function achieves subject-wise cross-validation.
        :param subjects: Subjects to load.
        :param shuffle: Whether to shuffle the training data.
        :param rand_state: See sklearn.model_selection._split.KFold
        :param rate: Rate at which to split the subjects.
        """
        if subjects is None:
            subjects = [0]
        tta = []  # total test accuracy
        tva = []  # total validation accuracy
        ttf = []  # total test f1
        tvf = []  # total validation f1

        for excluded_subs in subject_fold(subjects, rate):
            sub = excluded_subs[0]
            data_train, label_train = self.load_all_except_some(excluded_subs, shuffle=shuffle,
                                                                verbose=not self.args.reproduce)
            data_test, label_test = self.load_subjects(excluded_subs, verbose=not self.args.reproduce)
            va_val = Averager()
            vf_val = Averager()
            print('Subject fold: {} excluded'.format(', '.join([str(sub) for sub in excluded_subs])))
            data_train, label_train = self.prepare_data_subject_fold(data_train, label_train)
            data_test, label_test = self.prepare_data_subject_fold(data_test, label_test)

            data_train = np.expand_dims(data_train, axis=1)
            data_test = np.expand_dims(data_test, axis=1)
            acc_val = 0
            f1_val = 0
            if not self.args.reproduce:
                # to train new models
                acc_val, f1_val = self.first_stage(data=data_train, label=label_train, data_val=data_test,
                                                   label_val=label_test, subject=sub, fold=0,
                                                   rand_state=rand_state, phase=1)

                combine_train(args=self.args, data=data_train, label=label_train, data_val=data_test,
                              label_val=label_test, subject=sub, fold=0, phase=1)

            acc_test, f1, cm = test(args=self.args, data=data_test, label=label_test, reproduce=self.args.reproduce,
                                    subject=sub, phase=1)
            print("Confusion matrix ([[TN, FP], [FN, TP]]):\n", cm)
            if not self.args.reproduce:
                self.aggregate_compute_score(va_val, acc_val, vf_val, f1_val, tva, tvf, tta,
                                             ttf, acc_test, f1)
        if not self.args.reproduce:
            self.final_print(tta, ttf, tva, tvf)

    def subject_fold_cv_phase_2_3(self, subjects=None, phase: int = 2, rate: float = .2):
        """
        This function achieves phases 2 and 3 of subject-wise cross-validation.
        :param subjects: The subjects to load.
        :param phase: Which training phase to execute.
        :param rate: The percentage of subjects for the multiple-subject-wise cross-validation.
        """

        if subjects is None:
            subjects = [0]
        tta = []  # total test accuracy
        tva = []  # total validation accuracy
        ttf = []  # total test f1
        tvf = []  # total validation f1

        all_data, all_label = self.load_all(verbose=not self.args.reproduce, rate=rate)
        all_data_p, all_label_p, all_dataloaders = [], [], []
        for data, label in zip(all_data, all_label):
            data_p, label_p = self.prepare_data_subject_fold(data, label)
            data_p = np.expand_dims(data_p, axis=1)
            all_data_p.append(data_p)
            all_label_p.append(label_p)
            all_dataloaders.append(get_dataloader(data_p, label_p, self.args.batch_size))

        for excluded_subs in subject_fold(subjects, rate):
            print('Subject fold: {} excluded'.format(', '.join([str(sub) for sub in excluded_subs])))
            excluded_sub = excluded_subs[0]

            val_loaders = [all_dataloaders[exc] for exc in excluded_subs]

            acc_val = 0
            f1_val = 0
            patient = self.args.patient

            if not self.args.reproduce:

                model = get_RNNLGG(self.args, excluded_sub, phase=phase)
                lr = 0.001
                if phase == 2:
                    lr *= 2
                optimizer = optim.Adam(model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
                criterion = nn.BCELoss()
                va_val = Averager()
                vf_val = Averager()
                early_stopping = False

                timer = Timer()
                trlog = {'args': vars(self.args), 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
                         'max_acc': 0.0, 'F1': 0.0}
                counter = 0

                for epoch in range(1, self.args.phase_2_epochs + 1):
                    if early_stopping:
                        break
                    for sub in subjects:
                        if sub in excluded_subs:
                            continue

                        train_loader = all_dataloaders[sub]

                        tl = Averager()
                        pred_train = []
                        act_train = []
                        h_0 = None
                        optimizer.zero_grad()
                        for data, label in train_loader:
                            actual_batch_size = data.size(0)

                            if h_0 is not None and h_0.size(1) != actual_batch_size:
                                h_0 = torch.zeros(h_0.size(0), actual_batch_size, h_0.size(2), device=DEVICE)

                            data, label = data.to(DEVICE), label.to(DEVICE)
                            out, h_0 = model(data, h_0)
                            h_0 = h_0.detach()
                            pred = (out >= .5).int()
                            loss = criterion(out, label)

                            pred_train.extend(pred.data.tolist())
                            act_train.extend(label.data.tolist())

                            tl.add(loss.item())
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                        acc_train, f1_train = get_metrics(y_pred=pred_train, y_true=act_train, get_cm=False)
                        loss_val, acc_val, f1_val = predict_phase_2_3(data_loaders=val_loaders, net=model,
                                                                      loss_fn=criterion)
                        if epoch % 5 == 0 or epoch < 6:
                            print_cyan('[epoch {}] loss={:.4f} acc={:.4f} f1={:.4f}'
                                       .format(epoch, tl.item(), acc_train, f1_train))

                            print_purple(
                                '[epoch {}] (val) loss={:.4f} acc={:.4f} f1={:.4f}'.format(epoch, loss_val, acc_val,
                                                                                           f1_val))

                        if acc_val > trlog['max_acc'] and not np.isclose(acc_val, 1.):
                            trlog['max_acc'], trlog['F1'] = acc_val, f1_val
                            # save model here for reproduce
                            model_name_reproduce = 'sub{}_phase{}.pth'.format(excluded_sub, phase)
                            model_name_final = 'final_model_phase{}.pth'.format(
                                phase)  # Save final model here ? Not global save.
                            data_type = 'model'
                            experiment_setting = 'T_{}_pool_{}'.format(self.args.T, self.args.pool)
                            save_path = osp.join(self.args.save_path, experiment_setting, data_type)
                            ensure_path(save_path)
                            model_name_reproduce = osp.join(save_path, model_name_reproduce)
                            model_name_final = osp.join(self.args.save_path, model_name_final)
                            torch.save(model.state_dict(), model_name_reproduce)
                            torch.save(model.state_dict(), model_name_final)

                            counter = 0
                        else:
                            counter += 1
                            if counter >= patient * self.subjects // 5:
                                print_cyan('[epoch {}] loss={:.4f} acc={:.4f} f1={:.4f}'
                                           .format(epoch, tl.item(), acc_train, f1_train))
                                print_purple(
                                    '[epoch {}] (val) loss={:.4f} acc={:.4f} f1={:.4f}'.format(epoch, loss_val, acc_val,
                                                                                               f1_val))
                                print('ETA:{}/{} EXC_SUB:{} SUB:{}'.format(timer.measure(),
                                                                           timer.measure(
                                                                               epoch / self.args.phase_2_epochs),
                                                                           ', '.join(
                                                                               [str(sub) for sub in excluded_subs]),
                                                                           sub))
                                print_red('Early stopping')
                                early_stopping = True
                                break

                        trlog['train_loss'].append(tl.item())
                        trlog['train_acc'].append(acc_train)
                        trlog['val_loss'].append(loss_val)
                        trlog['val_acc'].append(acc_val)

                        if epoch % 5 == 0 or epoch < 6:
                            print('ETA:{}/{} EXC_SUB:{} SUB:{}'.format(timer.measure(),
                                                                       timer.measure(epoch / self.args.phase_2_epochs),
                                                                       ', '.join([str(sub) for sub in excluded_subs]),
                                                                       sub))
            acc_test, f1, cm = test_phase_2_3(args=self.args, test_loaders=val_loaders,
                                              reproduce=self.args.reproduce, subject=excluded_sub, phase=phase)
            print("Confusion matrix ([[TN, FP], [FN, TP]]):\n", cm)

            if not self.args.reproduce:
                self.aggregate_compute_score(va_val, acc_val, vf_val, f1_val, tva, tvf, tta, ttf, acc_test, f1)
        if not self.args.reproduce:
            self.final_print(tta, ttf, tva, tvf)

    @staticmethod
    def aggregate_compute_score(va_val, acc_val, vf_val, f1_val, tva, tvf, tta, ttf, acc, f1):
        va_val.add(acc_val)
        vf_val.add(f1_val)
        tva.append(va_val.item())
        tvf.append(vf_val.item())
        tta.append(acc)
        ttf.append(f1)

    def final_print(self, tta, ttf, tva, tvf):
        # prepare final report
        tta = np.array(tta)
        ttf = np.array(ttf)
        tva = np.array(tva)
        tvf = np.array(tvf)
        mACC = np.mean(tta)
        mF1 = np.mean(ttf)
        std = np.std(tta)
        mACC_val = np.mean(tva)
        std_val = np.std(tva)
        mF1_val = np.mean(tvf)
        print('Final: test mean ACC:{} std:{}'.format(mACC, std))
        print('Final: test mean F1:{}'.format(mF1))
        if not self.args.reproduce:
            print('Final: val mean ACC:{} std:{}'.format(mACC_val, std_val))
            print('Final: val mean F1:{}'.format(mF1_val))

    def first_stage(self, data, label, data_val, label_val, subject, fold, rand_state=None, phase: int = 1):
        """
        this function achieves n-fold-CV to:
            1. select hyperparameters on training data
            2. get the model for evaluation on testing data
        :param data: (segments, 1, channel, data)
        :param label: (segments,)
        :param data_val: Data used for validation
        :param label_val: Label associated to validation data
        :param subject: which subject the data belongs to
        :param fold: which fold the data belongs to
        :param rand_state: See sklearn.model_selection._split.KFold
        :param phase: The current training phase
        :return: mean validation accuracy
        """
        # use n-fold-CV to select hyperparameters on training data
        # save the best performance model and the corresponding acc for the second stage
        kf = KFold(n_splits=3, shuffle=True, random_state=rand_state)
        va = Averager()
        vf = Averager()
        va_item = []
        maxAcc = 0.0
        for i, (idx_train, idx_val) in enumerate(kf.split(data)):
            print('Inner 3-fold-CV Fold:{}'.format(i))
            data_train, label_train = data[idx_train], label[idx_train]
            # Validation set now outside from the train subjects to avoid overfitting
            # data_val, label_val = data[idx_val], label[idx_val]

            acc_val, F1_val = train(args=self.args,
                                    data_train=data_train,
                                    label_train=label_train,
                                    data_val=data_val,
                                    label_val=label_val,
                                    subject=subject,
                                    fold=fold, phase=phase)

            va.add(acc_val)
            vf.add(F1_val)
            va_item.append(acc_val)
            if acc_val >= maxAcc:
                maxAcc = acc_val
                # choose the model with higher val acc as the model to second stage
                if self.args.model_type == 'RNNLGGnet':
                    old_name = osp.join(self.args.save_path, 'candidate_phase{}.pth'.format(phase))
                    new_name = osp.join(self.args.save_path, 'max-acc_phase{}.pth'.format(phase))
                else: #elif self.args.model_type == 'resnet':
                    old_name = osp.join(self.args.save_path, 'candidate.pth')
                    new_name = osp.join(self.args.save_path, 'max-acc.pth')
                if os.path.exists(new_name):
                    os.remove(new_name)
                os.rename(old_name, new_name)
                print('New max ACC model saved, with the val ACC being:{}'.format(acc_val))

        mAcc = va.item()
        mF1 = vf.item()
        return mAcc, mF1

    def log2txt(self, content):
        """
        this function log the content to results.txt
        :param content: string, the content to log
        """
        # with open(self.text_file, 'a') as file:
        #    file.write(str(content) + '\n')

    def compare(self, subjects=None, data_test=None, label_test=None, rate: float = .2, phase:int=1):
        """
        this function achieves n-fold cross-validation
        :param subjects: how many subject to load
        :param data_test: (segments, 1, channel, data)
        :param label_test: (segments,)
        :param rate: how much to split the data
        :param phase: The phase to use (if applicable)
        """
        # Train and evaluate the model subject by subject

        if subjects is None:
            subjects = []
        tta = []  # total test accuracy
        ttf = []  # total test f1

        accuracies = []

        for excluded_subs in subject_fold(subjects, rate):
            sub = excluded_subs[0]
            data_test, label_test = self.load_subjects(excluded_subs, verbose=not self.args.reproduce)
            print('Subject fold: {} excluded'.format(', '.join([str(sub) for sub in excluded_subs])))
            data_test, label_test = self.prepare_data_subject_fold(data_test, label_test)

            data_test = np.expand_dims(data_test, axis=1)

            acc, f1, cm = test(args=self.args, data=data_test, label=label_test, reproduce=self.args.reproduce,
                               subject=sub, phase=phase)
            print("Confusion matrix ([[TN, FP], [FN, TP]]):\n", cm)

            tta.append(acc)
            ttf.append(f1)

        # prepare final report
        mACC = np.mean(tta)
        std = np.std(tta)

        return mACC, std, accuracies, data_test, label_test
