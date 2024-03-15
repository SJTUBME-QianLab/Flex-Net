from __future__ import print_function
import pandas as pd
import numpy as np
import copy
import random
import argparse
import os
from sklearn.model_selection import KFold
from joblib import Parallel, delayed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./../data/')
    parser.add_argument('--data_name', type=str, default='nat1')
    parser.add_argument('--rand', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()

    data_dict = {
        'nat1': 'nat1_0.8_',
        'nat2': 'nat2_0.8_',
    }
    data_name = data_dict[args.data_name]
    out_dir = os.path.join(args.data_root, 'nat', data_name).rstrip('/')
    if args.data_name == 'nat1':
        divide_nat1(out_dir, args.rand)
    elif args.data_name == 'nat2':
        divide_nat2(out_dir, args.rand)
    else:
        raise ValueError(args.data_name)


def divide_nat1(data_dir, rand):
    save_dir = os.path.join(data_dir, f'divide_{rand}')
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(data_dir, 'index_data_label_lost.csv'), header=None)

    data_folds = []
    data_folds.append(pd.concat([cut_data(df, 4, 3), cut_data(df, 6, 3)], axis=0))

    df_l1 = cut_data(df, 1, 3)
    df_l1 = df_l1.sample(frac=1, random_state=rand).reset_index(drop=True, inplace=False)
    df_l1_0, df_l1_r = separate_data_cl3(df_l1, 1, 58/89)
    data_folds.append(df_l1_0)

    df_l7 = cut_data(df, 7, 3)
    df_l7 = df_l7.sample(frac=1, random_state=rand).reset_index(drop=True, inplace=False)
    df_l7_0, df_l7_r = separate_data_cl3(df_l7, 7, 27/142)
    data_folds.append(pd.concat([df_l7_0, df_l1_r], axis=0))

    df_l7_r = df_l7_r.sample(frac=1, random_state=rand).reset_index(drop=True, inplace=False)
    data_folds.extend(list(separate_data_cl3(df_l7_r, 7, 0.5)))

    temp_set = [cut_data(df, ll, 3) for ll in [0, 2, 3, 5, 8]]
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rand)
    temp_ll = [list(skf.split(range(len(x)), x.iloc[:, -2])) for x in temp_set]

    for i, [_, test0], [_, test1], [_, test2], [_, test3], [_, test4] in \
            zip(range(5), temp_ll[0], temp_ll[1], temp_ll[2], temp_ll[3], temp_ll[4]):
            # zip(tuple([list(skf.split(range(len(x)), x.iloc[:, -2])) for x in temp_set])):
        ll = [yy.iloc[xx, :] for yy, xx in zip(temp_set, [test0, test1, test2, test3, test4])]
        new = pd.concat(ll, axis=0)
        data_folds[i] = pd.concat([new, data_folds[i]], axis=0)
    # --------------------------------------------------------------

    for i in range(5):
        outTe = os.path.join(save_dir, f'index_data_label_lost_{i}_test.csv')
        outTr = os.path.join(save_dir, f'index_data_label_lost_{i}_train.csv')
        if not (os.path.isfile(outTe) and os.path.getsize(outTe) > 0) or i == 4:
            test_data = data_folds[i]
            test_data_sc = my_scale(test_data)
            train_data = pd.concat([data_folds[i] for i in set(range(5)) - {i}], axis=0)
            train_data_sc = my_scale(train_data)
        if not (os.path.isfile(outTe) and os.path.getsize(outTe) > 0):
            test_data_sc.to_csv(outTe, index=False, header=None)
            train_data_sc.to_csv(outTr, index=False, header=None)
        if (os.path.isfile(outTe) and os.path.getsize(outTe) > 0) and i == 4:
            t1 = test_data_sc.reset_index(drop=True)
            t2 = pd.read_csv(outTe, header=None)
            pd.testing.assert_frame_equal(t1, t2)
            t1 = train_data_sc.reset_index(drop=True)
            t2 = pd.read_csv(outTr, header=None)
            pd.testing.assert_frame_equal(t1, t2)

        print('finish divide-scale-save data ' + str(i))


def divide_nat2(data_dir, rand):
    save_dir = os.path.join(data_dir, f'divide_{rand}')
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(data_dir, 'index_data_label_lost.csv'), header=None)
    lost_num = int(df.iloc[-1, -1]) + 1  # 20

    skf = KFold(n_splits=5, shuffle=True, random_state=rand)
    for i, [train, test] in zip(range(5), list(skf.split(range(lost_num)))):
        outTe = os.path.join(save_dir, f'index_data_label_lost_{i}_test.csv')
        outTr = os.path.join(save_dir, f'index_data_label_lost_{i}_train.csv')
        if not (os.path.isfile(outTe) and os.path.getsize(outTe) > 0) or i == 4:
            test_ind = df.iloc[np.where(df.iloc[:, -1].isin(test))[0], :].index.tolist()
            test_data = ex_data(df, test_ind)
            train_ind = df.iloc[np.where(df.iloc[:, -1].isin(train))[0], :].index.tolist()
            train_data = ex_data(df, train_ind)
        if not (os.path.isfile(outTe) and os.path.getsize(outTe) > 0):
            with open(os.path.join(save_dir, 'log.txt'), 'a') as f:
                f.write(str(i) + '\n')
                f.write(','.join([str(x) for x in test]) + '\n')
                f.write(','.join([str(x) for x in train]) + '\n')
            test_data.to_csv(outTe, index=False, header=None)
            train_data.to_csv(outTr, index=False, header=None)
        if (os.path.isfile(outTe) and os.path.getsize(outTe) > 0) and i == 4:
            t1 = test_data.reset_index(drop=True)
            t2 = pd.read_csv(outTe, header=None)
            pd.testing.assert_frame_equal(t1, t2)
            t1 = train_data.reset_index(drop=True)
            t2 = pd.read_csv(outTr, header=None)
            pd.testing.assert_frame_equal(t1, t2)

        print('finish divide-scale-save data ' + str(i))


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
    dfii.reset_index(drop=True, inplace=True)
    return dfii


def separate_data_cl3(df, lost, frac):
    df_sep = [cut_data(df, lost, i) for i in range(3)]
    df_1 = [dfi.iloc[:int(len(dfi) * frac), :] for dfi in df_sep]
    df_1 = pd.concat(df_1, axis=0)
    df_1.reset_index(drop=True, inplace=True)

    df_2 = [dfi.iloc[int(len(dfi) * frac):, :] for dfi in df_sep]
    df_2 = pd.concat(df_2, axis=0)
    df_2.reset_index(drop=True, inplace=True)

    return df_1, df_2


if __name__ == '__main__':
    main()
