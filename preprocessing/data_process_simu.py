from __future__ import print_function
import pandas as pd
import numpy as np
import copy
import random
import argparse
import os
import time
from sklearn.model_selection import KFold, StratifiedKFold
from scipy import optimize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./../data/')
    parser.add_argument('--mode', type=str, default='MCAR', choices=['MCAR', 'MAR', 'MNAR'])
    parser.add_argument('--rand', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--lost_num', type=int, default=33, help='Number of features with missing values')
    parser.add_argument('--p_inc', type=float, default=0., help='Proportion of incomplete samples for one feature')
    args = parser.parse_args()

    out_dir = os.path.join(args.data_root, 'simu', 'simu_1100').rstrip('/')
    if args.lost_num == 0:
        full(out_dir, args.rand)
    else:
        simulate(out_dir, args.rand, args.lost_num, args.p_inc, args.mode)


def full(data_root, rand):
    lost_kind = 50
    sample_num = 1100
    block = 22
    assert data_root.endswith(str(sample_num))
    save_dir = os.path.join(data_root + '_full', f'divide_{rand}')
    os.makedirs(save_dir, exist_ok=True)

    data_part = pd.read_csv(data_root + '.csv', header=None)
    data_part.loc[:, data_part.shape[1]] = sorted(list(range(1, lost_kind + 1)) * block)
    feature_num = data_part.shape[1] - 3
    assert feature_num == 663

    # complete. divide training / testing set (according to the loss_flag)
    if not os.path.isfile(os.path.join(save_dir, 'index_data_label_full.csv')):
        # sample(frac=1, random_state=rand).reset_index(drop=True, inplace=False)
        skf = KFold(n_splits=5, shuffle=True, random_state=rand)
        data_full = []
        for k, [train0, test0] in zip(range(5), list(skf.split(range(lost_kind)))):
            outTe = os.path.join(save_dir, f'index_data_label_full_{k}_test.csv')
            outTr = os.path.join(save_dir, f'index_data_label_full_{k}_train.csv')
            test_data = pd.concat([cut_data(data_part, tt + 1, 3) for tt in test0], axis=0)
            test_data = my_scale(test_data)
            train_data = pd.concat([cut_data(data_part, tt + 1, 3) for tt in set(range(lost_kind)) - set(test0)], axis=0)
            train_data = my_scale(train_data)
            test_data.to_csv(outTe, index=False, header=False)
            train_data.to_csv(outTr, index=False, header=False)
            data_full.append(test_data)

        data_full = pd.concat(data_full, axis=0)
        data_full.to_csv(os.path.join(save_dir, 'index_data_label_full.csv'), index=False, header=None)
        dfnew = pd.merge(data_full, data_part[0], on=0, how='outer')
        assert dfnew.shape[0] == data_part.shape[0]
        del dfnew
        print('finish full data')
    else:
        data_full = pd.read_csv(os.path.join(save_dir, 'index_data_label_full.csv'), header=None)
        assert data_full.shape[0] == sample_num
        print('full data already exist')

    # check full
    k = 0
    outTe = os.path.join(save_dir, f'index_data_label_full_{k}_test.csv')
    outTr = os.path.join(save_dir, f'index_data_label_full_{k}_train.csv')
    if os.path.isfile(outTe) and os.path.getsize(outTe) > 0:
        skf = KFold(n_splits=5, shuffle=True, random_state=rand)
        [train0, test0] = list(skf.split(range(lost_kind)))[0]
        test_data = pd.concat([cut_data(data_part, tt + 1, 3) for tt in test0], axis=0)
        test_data = my_scale(test_data)
        train_data = pd.concat([cut_data(data_part, tt + 1, 3) for tt in set(range(lost_kind)) - set(test0)], axis=0)
        train_data = my_scale(train_data)
        t1 = test_data.reset_index(drop=True)
        t2 = pd.read_csv(outTe, header=None)
        assert pd.testing.assert_frame_equal(t1, t2) is None
        t1 = train_data.reset_index(drop=True)
        t2 = pd.read_csv(outTr, header=None)
        assert pd.testing.assert_frame_equal(t1, t2) is None


def simulate(data_root, rand, lost_num, p_inc, mode):
    sample_num = 1100
    assert data_root.endswith(str(sample_num))
    save_dir = os.path.join(data_root + f'_{mode}Fix', f'divide_{rand}', 'L{:d}p{:g}'.format(lost_num, p_inc))
    full_dir = os.path.join(data_root + '_full', f'divide_{rand}')
    if lost_num > 0:
        os.makedirs(save_dir, exist_ok=True)

    RawData = pd.read_csv(data_root + '.csv', header=None)
    feature_num = RawData.shape[1] - 2
    assert feature_num == 663

    assert os.path.isfile(os.path.join(full_dir, 'index_data_label_full_4_test.csv'))

    data_full = pd.read_csv(os.path.join(full_dir, 'index_data_label_full.csv'), header=None)
    # generate the missing data using different missing mechanisms
    if not os.path.isfile(os.path.join(save_dir, 'index_data_label_lost.csv')):
        np.random.seed(rand)
        X_true = data_full.iloc[:, 1:-2].values
        if mode == 'MAR':
            mask = MAR_mask(X_true, lost_num, p_inc)  # rate~(lost_num/d)*p_inc
        elif mode == 'MNAR':
            mask = MNAR_mask(X_true, lost_num, p_inc)  # rate~p_inc
        elif mode == 'MCAR':
            mask = MCAR_mask(X_true, p_inc)  # rate~p_inc
        else:
            raise ValueError(mode)
        X_true[mask] = np.nan  # 1=missing, 0=observed
        data_full = pd.concat([data_full.iloc[:, 0], pd.DataFrame(X_true), data_full.iloc[:, -2]], axis=1)
        pd.DataFrame((1-mask).astype(int)).to_csv(os.path.join(save_dir, 'lost_pattern.csv'), index=False, header=None)
        print('finish missing data, incomplete proportion: ', mask.sum()/mask.size)

        # statistics for missing patterns
        data_loss = define_loss(data_full)
        missing_f = np.sum(mask.sum(axis=0) > 0)
        missing_s = np.sum(mask.sum(axis=1) > 0)
        data_loss.to_csv(os.path.join(save_dir, 'index_data_label_lost.csv'), index=False, header=None)

        with open(os.path.join(data_root + f'_{mode}Fix', 'log.txt'), 'a') as f:
            f.write("-------------------------------------\n")
            f.write(time.strftime("%Y-%m-%d %X", time.localtime()) + '\n')
            f.write('out dir:\t' + save_dir + '\n')
            f.write('loss pattern num:\t' + str(max(data_loss['loss_flag']) + 1) + '\n')
            f.write('sample num:\t' + str(data_loss.shape[0]) + '\n')
            f.write('feature num:\t' + str(data_loss.shape[1] - 3) + '\n')
            f.write('incomplete sample num:\t' + str(missing_s) + '\n')
            f.write('incomplete feature num:\t' + str(missing_f) + '\n')
            f.write('matrix incomplete rate:\t' + str(mask.sum()/mask.size) + '\n')

    # divide 5 folds
    print('full data exists, start lost data')
    data_part_lost = pd.read_csv(os.path.join(save_dir, 'index_data_label_lost.csv'), header=None)
    test_num = int(sample_num / 5)
    for i in range(5):
        outTe = os.path.join(save_dir, f'index_data_label_lost_{i}_test.csv')
        outTr = os.path.join(save_dir, f'index_data_label_lost_{i}_train.csv')
        if not (os.path.isfile(outTe) and os.path.getsize(outTe) > 0) or i == 4:
            test_ind = list(range(i * test_num, (i + 1) * test_num))
            test_data = ex_data(data_part_lost, test_ind)
            train_ind = list(set(range(sample_num)) - set(test_ind))
            train_data = ex_data(data_part_lost, train_ind)
        if not (os.path.isfile(outTe) and os.path.getsize(outTe) > 0):
            test_data.to_csv(outTe, index=False, header=False)
            train_data.to_csv(outTr, index=False, header=False)
            print('finish divide-scale-save missing data ' + str(i))
        if (os.path.isfile(outTe) and os.path.getsize(outTe) > 0) and i == 4:
            t1 = test_data.reset_index(drop=True)
            t2 = pd.read_csv(outTe, header=None)
            try:
                pd.testing.assert_frame_equal(t1, t2)
            except AssertionError:
                print(outTe)
            t1 = train_data.reset_index(drop=True)
            t2 = pd.read_csv(outTr, header=None)
            try:
                pd.testing.assert_frame_equal(t1, t2)
            except AssertionError:
                print(outTr)
            print('finish check missing data ' + str(i))


def ex_data(data, ind):
    data_part = data.iloc[ind, :]
    data_part_sc = my_scale(data_part)
    return data_part_sc


def my_scale(data):
    data_sc = data.iloc[:, 1:-2].apply(lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)), axis=0)
    data_sc = pd.concat([data.iloc[:, 0], data_sc, data.iloc[:, -2:]], axis=1)
    return data_sc


def cut_data(df, lost, label):
    dfi = df[df.iloc[:, -1] == lost]
    # print(dfi.shape)
    if label == 3:
        dfii = dfi
    else:
        dfii = dfi[dfi.iloc[:, -2] == label]
    # print(dfii.shape)
    return dfii


def define_loss(data_5):
    data_6 = data_5.copy()
    loss = []

    for i in range(data_6.shape[0]):
        flag = ''
        for j in range(data_6.shape[1]):
            if np.isnan(data_6.iloc[i, j]):  # Y=Yes, missing
                flag = flag + 'Y'
            else:  # N=No, not missing, observed
                flag = flag + 'N'
        loss.append(flag)  # denote sample missing pattern using a string

    loss_set = np.unique(loss)
    loss_flag = dict(zip(loss_set, range(len(loss_set))))
    # loss_flag: the index of different missing patterns
    data_6['loss_flag'] = [loss_flag[kk] for kk in loss]

    return data_6


# ################ Missing mechanism ###########################
# Reference: https://github.com/BorisMuzellec/MissingDataOT

def MAR_mask(X, d_na=None, p_inc=None):
    """
    Missing at random mechanism with a logistic masking model.
    A subset of variables with *no* missing values is randomly selected.
    The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.

    :param X: np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
    :param d_na: float
        Number of variables with missing values that will not be used for the logistic masking model.
    :param p_inc: float
        Proportion of missing values to generate for variables which will have missing values.
    :return: mask : np.ndarray
        Mask of generated missing values (True if the value is missing).
    """
    n, d = X.shape
    mask = np.zeros((n, d)).astype(bool)
    d_na = min(d_na, d)  # number of variables that will have missing values (at most d variable)
    d_obs = d - d_na  # number of variables that will have no missing values

    # Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    # Other variables will have NA proportions that depend on those observed variables, through a logistic model
    # The parameters of this logistic model are random.

    # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)  # X: [150,4]
    # Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p_inc)

    ps = sigmoid(np.matmul(X[:, idxs_obs], coeffs) + intercepts)

    ber = np.random.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps  # ps比例的缺失，mask=1表示缺失

    return mask


def MNAR_mask(X, d_na=None, p_inc=None):
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) d_na = d
    Missing probabilities are selected with a logistic model, taking all variables as inputs.
    Hence, values that are inputs can also be missing.
    (ii) d_na < d
    Variables are split into
    a set of intputs for a logistic model, and
    a set whose missing probabilities are determined by the logistic model.
    Then inputs are masked MCAR (hence, missing values from the second set will depend on masked values.

    The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.

    :param X: np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
    :param d_na: float
        Number of variables with missing values that will not be used for the logistic masking model.
    :param p_inc: float
        Proportion of missing values to generate for variables which will have missing values.
    :return: mask : np.ndarray
        Mask of generated missing values (True if the value is missing).
    """
    n, d = X.shape
    mask = np.zeros((n, d)).astype(bool)
    if d_na == d:
        idxs_params = np.arange(d)
        idxs_nas = np.arange(d)
    elif d_na < d:  # number of variables masked with the logistic model
        d_params = d - d_na  # number of variables used as inputs (at least 1)
        assert d_params >= 1
        # Sample variables that will be parameters for the logistic regression:
        idxs_params = np.random.choice(d, d_params, replace=False)
        idxs_nas = np.array([i for i in range(d) if i not in idxs_params])
    else:
        raise ValueError('d_na', d_na)

    # Other variables will have NA proportions that depend on those observed variables, through a logistic model
    # The parameters of this logistic model are random.

    # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_params, idxs_nas)  # X: [150,4]
    # Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p_inc)

    ps = sigmoid(np.matmul(X[:, idxs_params], coeffs) + intercepts)

    ber = np.random.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    # If the inputs of the logistic model are excluded from MNAR missingness,
    # mask some values used in the logistic model at random.
    # This makes the missingness of other variables potentially dependent on masked values
    if d_na < d:
        mask[:, idxs_params] = np.random.rand(n, d - d_na) < p_inc

    return mask


def MCAR_mask(X, p=None):
    return np.random.rand(*X.shape) < p


def pick_coeffs(X, idxs_obs=None, idxs_nas=None):
    coeffs = np.random.rand(len(idxs_obs), len(idxs_nas))
    Wx = np.matmul(X[:, idxs_obs], coeffs)
    coeffs /= np.std(Wx, 0, keepdims=True)
    return coeffs


def fit_intercepts(X, coeffs, p):
    d_obs, d_na = coeffs.shape
    intercepts = np.zeros(d_na)
    for j in range(d_na):  # 每个缺失特征
        def f(x):  # X: (X[:, idxs_obs]) [n,d_obs], coeffs[:, j]: [d_obs, 1], mv: [n, 1]
            return sigmoid(np.matmul(X, coeffs[:, j]) + x).mean() - p  # mv: matrix-vector
        intercepts[j] = optimize.bisect(f, -50, 50)  # Find the zero point of the function f in the interval [a,b]
    return intercepts


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    main()
