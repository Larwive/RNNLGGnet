import copy
from sklearn.model_selection import KFold
from functools import reduce
from train import *
from utils import Averager, ensure_path

ROOT = os.getcwd()


class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        # Log the results per subject
        result_path = osp.join(args.save_path, 'result')
        ensure_path(result_path)
        self.text_file = osp.join(result_path,
                                  "results_{}.txt".format(args.dataset))
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
        sub_code = 'sub{}.hdf'.format(sub)
        path = osp.join(save_path, data_type, sub_code)
        with h5py.File(path, 'r') as dataset:
            data = np.array(dataset['data'])
            label = np.array(dataset['label'])
        return data, label

    def load_per_subject(self, sub: int):
        """
        load data for sub
        :param sub: which subject's data to load
        :return: data and label
        """
        save_path = os.getcwd()
        data_type = 'data_{}'.format(self.args.dataset)
        data, label = self.read_data(sub, save_path, data_type)
        print('>>> Data:{} Label:{}'.format(data.shape, label.shape))
        return data, label

    def load_all_except_one(self, excluded_sub: int, max_subject: int = 27, shuffle: bool = False):
        save_path = os.getcwd()
        data_type = 'data_{}'.format(self.args.dataset)
        datas, labels = [], []
        for sub in range(max_subject):
            if sub == excluded_sub:
                continue
            data, label = self.read_data(sub, save_path, data_type)
            datas.append(data)
            labels.append(label)
            print('>>> Data:{} Label:{}'.format(datas[-1].shape, labels[-1].shape))
        datas = reduce(lambda x, y: np.concatenate((x, y), axis=0), datas)
        labels = reduce(lambda x, y: np.concatenate((x, y), axis=0), labels)
        if shuffle:
            permutation = np.random.permutation(len(datas))
            datas = datas[permutation]
            labels = labels[permutation]

        return datas, labels

    def load_all(self, max_subject: int = 27):
        save_path = os.getcwd()
        data_type = 'data_{}'.format(self.args.dataset)
        datas, labels = [], []
        for sub in range(max_subject):
            data, label = self.read_data(sub, save_path, data_type)
            datas.append(data)
            labels.append(label)
            print('>>> Data:{} Label:{}'.format(datas[-1].shape, labels[-1].shape))

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
        return self.normalize_data(data_train, label_train, data_test, label_test)

    def prepare_data_subject_fold(self, train_data, train_label, test_data, test_label):
        """

        :param train_data:
        :param train_label:
        :param test_data:
        :param test_label:
        :return:
        """

        """
        We want to do subject-wise fold, so the idx_train/idx_test is for
        trials.
        data: (trial, segment, 1, chan, datapoint)
        To use the normalization function, we should change the dimension from
        (trial, segment, 1, chan, datapoint) to (trial*segments, 1, chan, datapoint)
        """
        data_train = np.concatenate(train_data, axis=0)
        label_train = np.concatenate(train_label, axis=0)
        data_test = test_data
        label_test = test_label
        return self.normalize_data(data_train, label_train, data_test, label_test)

    def normalize_data(self, data_train, label_train, data_test, label_test):
        if len(data_test.shape) > 4:
            """
            When leave one trial out is conducted, the test data will be (segments, 1, chan, datapoint), hence,
            no need to concatenate the first dimension to get trial*segments
            """
            data_test = np.concatenate(data_test, axis=0)
            label_test = np.concatenate(label_test, axis=0)
        data_train, data_test = self.normalize(train_data=data_train, test_data=data_test)
        # Prepare the data format for training the model using PyTorch
        data_train = torch.from_numpy(data_train).float()
        label_train = torch.from_numpy(label_train).long()
        data_test = torch.from_numpy(data_test).float()
        label_test = torch.from_numpy(label_test).long()
        return data_train, label_train, data_test, label_test

    @staticmethod
    def normalize(train_data, test_data):
        """
        this function do standard normalization for EEG channel by channel
        TTTTT Probably useless because data is already subject-wise normalized
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
        # Data dimension: segment x 1 x channel x data
        # Label dimension: segment x 1
        np.random.seed(0)
        # data : segments x 1 x channel x data
        # label : segments

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

    def subject_fold_CV(self, subject=None, shuffle=False, rand_state=None):
        """
        this function achieves n-fold cross-validation
        :param subject: how many subject to load
        :param shuffle:
        :param rand_state:
        """
        # Train and evaluate the model subject by subject
        if subject is None:
            subject = [0]
        tta = []  # total test accuracy
        tva = []  # total validation accuracy
        ttf = []  # total test f1
        tvf = []  # total validation f1

        for sub in subject:
            data_train, label_train = self.load_all_except_one(sub, shuffle=shuffle)
            data_test, label_test = self.load_per_subject(sub)
            va_val = Averager()
            vf_val = Averager()
            preds, acts = [], []
            print('Subject fold: {} excluded'.format(sub))
            data_train, label_train, data_test, label_test = self.prepare_data_subject_fold(
                train_data=data_train, train_label=label_train, test_data=data_test, test_label=label_test)

            if self.args.reproduce:
                acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                           reproduce=self.args.reproduce,
                                           subject=sub, fold=0)
                acc_val = 0
                f1_val = 0
            else:
                # to train new models
                acc_val, f1_val = self.first_stage(data=data_train, label=label_train,
                                                   subject=sub, fold=0, rand_state=rand_state)

                combine_train(args=self.args,
                              data=data_train, label=label_train,
                              subject=sub, fold=0, target_acc=1)

                acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                           reproduce=self.args.reproduce,
                                           subject=sub, fold=0)
            self.aggregate_compute_score(va_val, acc_val, vf_val, f1_val, preds, pred, acts, act, tva, tvf, tta,
                                         ttf)

        self.final_print(tta, tva, tvf)

    def subject_fold_cv_phase_2_3(self, subject=None, phase: int = 2):
        """
        this function achieves n-fold cross-validation
        :param subject: how many subject to load
        :param phase:
        """

        def save_model(name):
            previous_model = osp.join(self.args.save_path, 'RNN_LGG_{}.pth'.format(name))
            if os.path.exists(previous_model):
                os.remove(previous_model)
            torch.save(model.state_dict(), osp.join(self.args.save_path, 'RNN_LGG_{}.pth'.format(name)))

        # Train and evaluate the model subject by subject
        if subject is None:
            subject = [0]
        tta = []  # total test accuracy
        tva = []  # total validation accuracy
        ttf = []  # total test f1
        tvf = []  # total validation f1

        data, label = self.load_all()

        for excluded_sub in subject:
            data_test = data[excluded_sub]
            label_test = label[excluded_sub]
            val_loader = get_dataloader(data_test, label_test, self.args.batch_size)
            model = get_RNNLGG(self.args, excluded_sub, phase)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
            criterion = nn.BCELoss()
            for sub in subject:
                if sub == excluded_sub:
                    continue

                model.zero_grad()
                optimizer.zero_grad()

                data_train = data[sub]
                label_train = label[sub]

                va_val = Averager()
                vf_val = Averager()
                preds, acts = [], []
                print('Subject fold: {} excluded'.format(sub))
                data_train, label_train, data_test, label_test = self.prepare_data_subject_fold(
                    train_data=data_train, train_label=label_train, test_data=data_test, test_label=label_test)
                train_loader = get_dataloader(data_train, label_train, batch_size=self.args.batch_size)
                acc_val = 0
                f1_val = 0
                pred, act = None, None
                if self.args.reproduce:
                    acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                               reproduce=self.args.reproduce,
                                               subject=sub, fold=0, phase=phase)

                else:
                    trlog = {'args': vars(self.args), 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
                             'max_acc': 0.0,
                             'F1': 0.0}
                    timer = Timer()
                    patient = self.args.patient
                    counter = 0

                    for epoch in range(self.args.phase_2_epochs):
                        tl = Averager()
                        pred_train = []
                        act_train = []
                        for data, label in train_loader:
                            out = model(data)
                            pred_train.extend(out.data.tolist())
                            act_train.extend(label.data.tolist())

                            loss = criterion(out, label)
                            tl.add(loss.item())
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
                        print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
                              .format(epoch, tl.item(), acc_train, f1_train))

                        loss_val, pred_val, act_val = predict(data_loader=val_loader, net=model, loss_fn=criterion)
                        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
                        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.
                              format(epoch, loss_val, acc_val, f1_val))

                        if acc_val >= trlog['max_acc']:
                            trlog['max_acc'] = acc_val
                            trlog['F1'] = f1_val
                            save_model('candidate')
                            counter = 0
                        else:
                            counter += 1
                            if counter >= patient:
                                print('early stopping')
                                break

                        trlog['train_loss'].append(tl.item())
                        trlog['train_acc'].append(acc_train)
                        trlog['val_loss'].append(loss_val)
                        trlog['val_acc'].append(acc_val)

                        print(
                            'ETA:{}/{} SUB:{}'.format(timer.measure(), timer.measure(epoch / self.args.phase_2_epochs),
                                                      subject))

                self.aggregate_compute_score(va_val, acc_val, vf_val, f1_val, preds, pred, acts, act, tva, tvf, tta,
                                             ttf)

        self.final_print(tta, tva, tvf)

    @staticmethod
    def aggregate_compute_score(va_val, acc_val, vf_val, f1_val, preds, pred, acts, act, tva, tvf, tta, ttf):
        va_val.add(acc_val)
        vf_val.add(f1_val)
        preds.extend(pred)
        acts.extend(act)
        tva.append(va_val.item())
        tvf.append(vf_val.item())
        acc, f1, _ = get_metrics(y_pred=preds, y_true=acts)
        tta.append(acc)
        ttf.append(f1)

    @staticmethod
    def final_print(tta, tva, tvf):
        # prepare final report
        tta = np.array(tta)
        # ttf = np.array(ttf)
        tva = np.array(tva)
        tvf = np.array(tvf)
        mACC = np.mean(tta)
        # mF1 = np.mean(ttf)
        std = np.std(tta)
        mACC_val = np.mean(tva)
        std_val = np.std(tva)
        mF1_val = np.mean(tvf)
        print('Final: test mean ACC:{} std:{}'.format(mACC, std))
        print('Final: val mean ACC:{} std:{}'.format(mACC_val, std_val))
        print('Final: val mean F1:{}'.format(mF1_val))

    def first_stage(self, data, label, subject, fold, rand_state=None):
        """
        this function achieves n-fold-CV to:
            1. select hyperparameters on training data
            2. get the model for evaluation on testing data
        :param data: (segments, 1, channel, data)
        :param label: (segments,)
        :param subject: which subject the data belongs to
        :param fold: which fold the data belongs to
        :param rand_state: See sklearn.model_selection._split.KFold
        :return: mean validation accuracy
        """
        # use n-fold-CV to select hyperparameters on training data
        # save the best performance model and the corresponding acc for the second stage
        # data: trial x 1 x channel x time
        kf = KFold(n_splits=3, shuffle=True, random_state=rand_state)
        va = Averager()
        vf = Averager()
        va_item = []
        maxAcc = 0.0
        for i, (idx_train, idx_val) in enumerate(kf.split(data)):
            print('Inner 3-fold-CV Fold:{}'.format(i))
            data_train, label_train = data[idx_train], label[idx_train]
            data_val, label_val = data[idx_val], label[idx_val]
            acc_val, F1_val = train(args=self.args,
                                    data_train=data_train,
                                    label_train=label_train,
                                    data_val=data_val,
                                    label_val=label_val,
                                    subject=subject,
                                    fold=fold)

            va.add(acc_val)
            vf.add(F1_val)
            va_item.append(acc_val)
            if acc_val >= maxAcc:
                maxAcc = acc_val
                # choose the model with higher val acc as the model to second stage
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
        with open(self.text_file, 'a') as file:
            file.write(str(content) + '\n')
