from __future__ import print_function
import torch.utils.data as data
import torch
import pandas as pd
import numpy as np
import random
import os


class Load(data.Dataset):
    def __init__(self, data_path, seed, fold, partition='train'):
        self.partition = partition  # training set or test set

        rawdata = pd.read_csv(os.path.join(data_path, f'divide_{seed}',
                                           f'index_data_label_lost_{fold}_{partition}.csv'), header=None)

        self.feature_num = rawdata.shape[1] - 3
        self.sample_num = rawdata.shape[0]
        self.data = rawdata.iloc[:, 1: 1 + self.feature_num]
        # The first column is sample index, and the last two columns are label and missing pattern index.
        self.class0 = rawdata.iloc[:, -1]  # Missing patterns of all samples
        self.classes_ = set(self.class0)  # Set of missing patterns
        self.label = rawdata.iloc[:, -2].values.astype(int)  # class label of all samples
        self.label_num = len(set(self.label))  # 2 or 3

        rawdata.columns = ['index'] + [str(i + 1) for i in range(rawdata.shape[1] - 3)] + ['label', 'loss_flag']
        sub0 = rawdata[rawdata['label'] == 0]
        sub1 = rawdata[rawdata['label'] == 1]
        self.alter = [
            kk for kk in self.classes_ if
            (len(sub0[sub0['loss_flag'] == kk]) >= 2 and len(sub1[sub1['loss_flag'] == kk]) >= 2)
        ]

    def get_task_batch_eq(self, batch_size=20, n_way=5, num_shots=1):
        assert self.partition == 'train'
        batch_x = np.zeros((batch_size, self.feature_num), dtype='float32')  # 0(20,28)
        labels_x = np.zeros((batch_size, self.label_num), dtype='float32')  # 0(20,2)
        labels_x_global = np.zeros(batch_size, dtype='int64')  # 0(20)
        target_distances = np.zeros((batch_size, n_way * num_shots * self.label_num), dtype='float32')  # 0(20,5*1)
        batches_xi, labels_yi = [], []
        for i in range(n_way * num_shots * self.label_num):  # 5*1 rows
            batches_xi.append(np.zeros((batch_size, self.feature_num), dtype='float32'))  # [5](20,28)
            labels_yi.append(np.zeros((batch_size, self.label_num), dtype='float32'))  # [5](20,2)

        # Iterate over tasks for the same batch
        for batch_counter in range(batch_size):
            positive_class = random.randint(0, n_way - 1)

            # Sample random missing patterns for this TASK
            sampled_classes = random.sample(list(set(self.class0)), n_way)
            indexes_perm = np.random.permutation(n_way * num_shots * self.label_num)

            counter = 0  # number of samples
            for class_counter, class_ in enumerate(sampled_classes):
                this_class = np.where(self.class0 == class_)[0]
                this_class_0 = this_class[np.where(self.label[this_class] == 0)[0]]
                this_class_1 = this_class[np.where(self.label[this_class] == 1)[0]]
                this_class = this_class.tolist()
                if len(this_class_0) <= 1:
                    class_ = random.sample(list(set(self.alter)), 1)[0]
                    this_class_tmp = np.where(self.class0 == class_)[0]
                    this_class_0 = this_class_tmp[np.where(self.label[this_class_tmp] == 0)[0]]
                    this_class.extend(this_class_0.tolist())
                if len(this_class_1) <= 1:
                    class_ = random.sample(list(set(self.alter)), 1)[0]
                    this_class_tmp = np.where(self.class0 == class_)[0]
                    this_class_1 = this_class_tmp[np.where(self.label[this_class_tmp] == 1)[0]]
                    this_class.extend(this_class_1.tolist())
                # label = self.label[this_class]
                if class_counter == positive_class:
                    # We take num_shots + one sample for one class
                    idx0 = random.sample(this_class_0.tolist(), num_shots)
                    idx1 = random.sample(this_class_1.tolist(), num_shots)
                    idx = random.sample(list(set(this_class) - set(idx0) - set(idx1)), 1)
                    samples = self.data.iloc[idx + idx0 + idx1, :].values
                    labels = self.label[idx + idx0 + idx1]
                    # Test sample is loaded
                    batch_x[batch_counter, :] = samples[0]  # first sample, to be classified, is put into batch_X.
                    labels_x[batch_counter, labels[0]] = 1  # label onehot of this sample
                    labels_x_global[batch_counter] = labels[0]  # label of this sample
                    samples = samples[1::]
                    labels = labels[1::]
                else:
                    idx0 = random.sample(this_class_0.tolist(), num_shots)
                    idx1 = random.sample(this_class_1.tolist(), num_shots)
                    samples = self.data.iloc[idx0 + idx1, :].values
                    labels = self.label[idx0 + idx1]

                for s_i in range(0, len(samples)):  # 'shots' number of samples for query
                    batches_xi[indexes_perm[counter]][batch_counter, :] = samples[s_i]
                    labels_yi[indexes_perm[counter]][batch_counter, labels[s_i]] = 1
                    target_distances[batch_counter, indexes_perm[counter]] = 0
                    counter += 1

        batch_x, batches_xi = self.fill_na_tgt(batch_x, batches_xi)

        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]

        labels_x_scalar = np.argmax(labels_x, 1)

        return_arr = [torch.from_numpy(batch_x), torch.from_numpy(labels_x), torch.from_numpy(labels_x_scalar),
                      torch.from_numpy(labels_x_global), batches_xi, labels_yi]

        return return_arr

    def get_whole_eq(self, batch_size=10, n_way=5, num_shots=1, set0=0):
        if (self.sample_num - set0) < batch_size:
            batch_size = self.sample_num - set0
            print('test to tail')
            print('batch_size: %d' % batch_size)
        if batch_size == 0:
            return None, 0
        batch_x = np.zeros((batch_size, self.feature_num), dtype='float32')  # 0(20,28)
        labels_x = np.zeros((batch_size, self.label_num), dtype='float32')  # 0(20,2)
        labels_x_global = np.zeros(batch_size, dtype='int64')  # 0(20)
        target_distances = np.zeros((batch_size, n_way * num_shots * self.label_num), dtype='float32')  # 0(20,5*1)
        # numeric_labels = []
        batches_xi, labels_yi = [], []
        for i in range(n_way * num_shots * self.label_num):
            batches_xi.append(np.zeros((batch_size, self.feature_num), dtype='float32'))  # [5](20,28)
            labels_yi.append(np.zeros((batch_size, self.label_num), dtype='float32'))  # [5](20,2)

        # Iterate over tasks for the same batch
        for batch_counter in range(batch_size):  # 20
            seti = set0 + batch_counter  # index of the sample to be tested
            set_la = self.class0[seti]  # missing pattern index of the sample to be tested
            # Sample random missing patterns for this TASK
            sampled_classes = random.sample(list(self.classes_ - {set_la}), n_way - 1) + [set_la]
            indexes_perm = np.random.permutation(n_way * num_shots * self.label_num)

            counter = 0  # number of samples
            for class_counter, class_ in enumerate(sampled_classes):
                this_class = np.where(self.class0 == class_)[0]
                if class_ == set_la:  # when meeting the sample to be tested
                    this_class = np.setdiff1d(this_class, seti)  # remove the test sample itself.
                    this_class_0 = this_class[np.where(self.label[this_class] == 0)[0]]
                    this_class_1 = this_class[np.where(self.label[this_class] == 1)[0]]
                    if len(this_class_0) == 0:
                        class_ = random.sample(list(set(self.alter)), 1)[0]
                        this_class_tmp = np.where(self.class0 == class_)[0]
                        this_class_0 = this_class_tmp[np.where(self.label[this_class_tmp] == 0)[0]]
                    if len(this_class_1) == 0:
                        class_ = random.sample(list(set(self.alter)), 1)[0]
                        this_class_tmp = np.where(self.class0 == class_)[0]
                        this_class_1 = this_class_tmp[np.where(self.label[this_class_tmp] == 1)[0]]

                    # We take num_shots + one sample for one class
                    idx0 = random.sample(this_class_0.tolist(), num_shots)
                    idx1 = random.sample(this_class_1.tolist(), num_shots)
                    samples = self.data.iloc[[seti] + idx0 + idx1, :].values
                    labels = self.label[[seti] + idx0 + idx1]
                    # Test sample is loaded
                    batch_x[batch_counter, :] = samples[0]  # first sample, to be classified, is put into batch_X.
                    labels_x[batch_counter, labels[0]] = 1  # label onehot of this sample
                    labels_x_global[batch_counter] = labels[0]  # label of this sample
                    samples = samples[1::]
                    labels = labels[1::]
                else:
                    this_class_0 = this_class[np.where(self.label[this_class] == 0)[0]]
                    this_class_1 = this_class[np.where(self.label[this_class] == 1)[0]]
                    if len(this_class_0) <= 1 or len(this_class_1) <= 1:
                        class_ = random.sample(list(set(self.alter)), 1)[0]
                        this_class = np.where(self.class0 == class_)[0]
                        this_class_0 = this_class[np.where(self.label[this_class] == 0)[0]]
                        this_class_1 = this_class[np.where(self.label[this_class] == 1)[0]]

                    idx0 = random.sample(this_class_0.tolist(), num_shots)
                    idx1 = random.sample(this_class_1.tolist(), num_shots)
                    samples = self.data.iloc[idx0 + idx1, :].values
                    labels = self.label[idx0 + idx1]

                for s_i in range(0, len(samples)):  # 'shots' number of samples for query
                    batches_xi[indexes_perm[counter]][batch_counter, :] = samples[s_i]
                    labels_yi[indexes_perm[counter]][batch_counter, labels[s_i]] = 1
                    target_distances[batch_counter, indexes_perm[counter]] = 0
                    counter += 1

        batch_x, batches_xi = self.fill_na_tgt(batch_x, batches_xi)

        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]

        labels_x_scalar = np.argmax(labels_x, 1)

        return_arr = [torch.from_numpy(batch_x), torch.from_numpy(labels_x), torch.from_numpy(labels_x_scalar),
                      torch.from_numpy(labels_x_global), batches_xi, labels_yi]

        return return_arr, seti

    def fill_na_tgt(self, test, train):
        train.extend([test])
        for k in range(len(train)):
            for i in range(train[k].shape[0]):
                for j in range(train[k].shape[1]):
                    if np.isnan(train[k][i, j]):
                        # train[k][i, j] = np.random.random(1) * 2 - 1
                        train[k][i, j] = np.random.random(1)
        test = train[-1]
        train = train[:-1]

        return test, train
