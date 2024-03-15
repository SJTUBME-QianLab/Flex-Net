from __future__ import print_function
import pandas as pd
import numpy as np
import copy
import random
import argparse
import os
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--rand', type=int, default=1, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()

    data_path = os.path.join(args.data_root, 'uci', args.data_name)
    divide(data_path, rand=args.rand)


def divide(data_dir, rand):
    save_dir = os.path.join(data_dir, f'divide_{rand}')
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(data_dir, 'index_data_label_lost.csv'), header=None)
    df.columns = ['index'] + [str(i + 1) for i in range(df.shape[1] - 3)] + ['label', 'loss_flag']

    data_folds = []
    # --------------------------------------------------------------
    if os.path.basename(data_dir) == 'BC':
        df_l0 = cut_data(df, 0, 2)
        df_l0 = df_l0.sample(frac=1, random_state=rand).reset_index(drop=True, inplace=False)
        df_l0_1, df_l0_r = separate_data_cl2(df_l0, 0, (699 / 5 - 16) / 683)
        data_folds.append(pd.concat([cut_data(df, 1, 2), df_l0_1], axis=0))
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=rand)
        for [train0, test0] in list(skf.split(range(len(df_l0_r)), df_l0_r.iloc[:, -2])):
            df_test0 = df_l0_r.iloc[test0, :]
            data_folds.append(df_test0)

    elif os.path.basename(data_dir) == 'CC':
        df_l1 = cut_data(df, 1, 2)

        remain = pd.concat([cut_data(df, kk, 2) for kk in set(df['loss_flag']) - {1}],
                           axis=0).reset_index(drop=True)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rand)
        for i, (_, te1), (_, te2) in zip(range(5), skf.split(range(len(remain)), remain.iloc[:, -2]),
                                      skf.split(range(len(df_l1)), df_l1.iloc[:, -2])):
            data_folds.append(pd.concat([remain.iloc[te1, :], df_l1.iloc[te2, :]], axis=0))
            print(len(data_folds[i]), set(data_folds[i]['loss_flag']))

    elif os.path.basename(data_dir) == 'CK':
        set1 = pd.concat([cut_data(df, kk, 2) for kk in [5, 10, 11, 58, 71]],
                           axis=0).reset_index(drop=True)
        remain = pd.concat([cut_data(df, kk, 2) for kk in set(df['loss_flag']) - {0, 5, 10, 11, 58, 71}],
                           axis=0).reset_index(drop=True)
        skf_loss = GroupKFold(n_splits=5)
        for (_, te1), (_, te2) in zip(skf_loss.split(range(len(set1)), groups=set1['loss_flag']),
                                      skf_loss.split(range(len(remain)), groups=remain['loss_flag'])):
            data_folds.append(pd.concat([set1.iloc[te1, :], remain.iloc[te2, :]], axis=0))
        df_l0 = cut_data(df, 0, 2)  # 158
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rand)
        for i, (train0, test0) in zip(range(5), skf.split(range(len(df_l0)), df_l0.iloc[:, -2])):
            df_test0 = df_l0.iloc[test0, :]
            tmp = data_folds[i]
            data_folds[i] = pd.concat([tmp, df_test0], axis=0)
            print(len(data_folds[i]), set(data_folds[i]['loss_flag']))

    elif os.path.basename(data_dir) == 'HC':
        set1 = pd.concat([cut_data(df, kk, 2) for kk in [11, 12, 13, 43, 45, 129, 145, 222]],
                           axis=0).reset_index(drop=True)
        remain = pd.concat([cut_data(df, kk, 2) for kk in set(df['loss_flag']) - {11, 12, 13, 43, 45, 129, 145, 222}],
                           axis=0).reset_index(drop=True)
        set1 = set1.sample(frac=1, random_state=rand).reset_index(drop=True, inplace=False)
        remain = remain.sample(frac=1, random_state=rand).reset_index(drop=True, inplace=False)
        skf_loss = GroupKFold(n_splits=5)
        for (_, te1), (_, te2) in zip(skf_loss.split(range(len(set1)), groups=set1['loss_flag']),
                                      skf_loss.split(range(len(remain)), groups=remain['loss_flag'])):
            data_folds.append(pd.concat([set1.iloc[te1, :], remain.iloc[te2, :]], axis=0))

    elif os.path.basename(data_dir) == 'HD':
        df_l0 = cut_data(df, 3, 2)
        df_l0 = df_l0.sample(frac=1, random_state=rand).reset_index(drop=True, inplace=False)
        df_l0_1, df_l0_5 = separate_data_cl2(df_l0, 3, 10/84)  # 9, 75

        set1 = pd.concat([cut_data(df, kk, 2) for kk in [2, 5, 12, 15]] + [df_l0_1],
                           axis=0).reset_index(drop=True)
        remain = pd.concat([cut_data(df, kk, 2) for kk in set(df['loss_flag']) - {3, 6, 2, 5, 12, 15}],
                           axis=0).reset_index(drop=True)
        skf_loss = GroupKFold(n_splits=5)
        for (_, te1), (_, te2) in zip(skf_loss.split(range(len(set1)), groups=set1['loss_flag']),
                                      skf_loss.split(range(len(remain)), groups=remain['loss_flag'])):
            data_folds.append(pd.concat([set1.iloc[te1, :], remain.iloc[te2, :]], axis=0))

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rand)
        for i, (train0, test0) in zip(range(5), skf.split(range(len(df_l0_5)), df_l0_5.iloc[:, -2])):
            df_test0 = df_l0_5.iloc[test0, :]
            tmp = data_folds[i]
            data_folds[i] = pd.concat([tmp, df_test0], axis=0)
        df_l0 = cut_data(df, 6, 2)
        for i, (train0, test0) in zip(range(5), skf.split(range(len(df_l0)), df_l0.iloc[:, -2])):
            df_test0 = df_l0.iloc[test0, :]
            tmp = data_folds[i]
            data_folds[i] = pd.concat([tmp, df_test0], axis=0)
            print(len(data_folds[i]), set(data_folds[i]['loss_flag']))

    elif os.path.basename(data_dir) == 'HP':
        df_l0 = cut_data(df, 1, 2)
        df_l0 = df_l0.sample(frac=1, random_state=rand).reset_index(drop=True, inplace=False)
        df_l01, df_l02 = separate_data_cl2(df_l0, 1, 0.5)  # 16, 16

        remain = pd.concat([cut_data(df, kk, 2) for kk in set(df['loss_flag']) - {0, 1, 3, 4, 5}],
                           axis=0).reset_index(drop=True)
        skf_loss = GroupKFold(n_splits=5)
        for (_, te1) in skf_loss.split(range(len(remain)), groups=remain['loss_flag']):
            data_folds.append(remain.iloc[te1, :])

        random.seed(rand)
        pack = random.sample(range(5), 5)
        for i, ll in zip(pack, [3, 4, 5, 0, 1]):
            tmp = data_folds[i]
            if ll == 0:
                data_folds[i] = pd.concat([tmp, df_l01], axis=0)
            elif ll == 1:
                data_folds[i] = pd.concat([tmp, df_l02], axis=0)
            else:
                data_folds[i] = pd.concat([tmp, cut_data(df, ll, 2)], axis=0)

        df_l0 = cut_data(df, 0, 2)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rand)
        for i, (train0, test0) in zip(range(5), skf.split(range(len(df_l0)), df_l0.iloc[:, -2])):
            df_test0 = df_l0.iloc[test0, :]
            tmp = data_folds[i]
            data_folds[i] = pd.concat([tmp, df_test0], axis=0)
            print(len(data_folds[i]), set(data_folds[i]['loss_flag']))

    elif os.path.basename(data_dir) == 'HS':
        remain = pd.concat([cut_data(df, kk, 2) for kk in set(df['loss_flag']) - {0, 1, 2, 14}],
                           axis=0).reset_index(drop=True)
        skf_loss = GroupKFold(n_splits=5)
        for (_, te1) in skf_loss.split(range(len(remain)), groups=remain['loss_flag']):
            data_folds.append(remain.iloc[te1, :])

        df_l0 = cut_data(df, 0, 2)
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=rand)
        for i, (train0, test0) in zip(range(2), skf.split(range(len(df_l0)), df_l0.iloc[:, -2])):
            df_test0 = df_l0.iloc[test0, :]
            tmp = data_folds[i]
            data_folds[i] = pd.concat([tmp, df_test0], axis=0)
            print(len(data_folds[i]), set(data_folds[i]['loss_flag']))
        tmp = data_folds[2]
        data_folds[2] = pd.concat([tmp, cut_data(df, 1, 2)], axis=0)
        tmp = data_folds[3]
        data_folds[3] = pd.concat([tmp, cut_data(df, 2, 2)], axis=0)
        tmp = data_folds[4]
        data_folds[4] = pd.concat([tmp, cut_data(df, 14, 2)], axis=0)

    elif os.path.basename(data_dir) == 'PI':
        df_l0 = cut_data(df, 0, 2)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rand)
        for i, (train0, test0) in zip(range(5), skf.split(range(len(df_l0)), df_l0.iloc[:, -2])):
            data_folds.append(df_l0.iloc[test0, :])

        remain = pd.concat([cut_data(df, kk, 2) for kk in set(df['loss_flag']) - {0}],
                           axis=0).reset_index(drop=True)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rand)
        for i, (train0, test0) in zip(range(5), skf.split(range(len(remain)), remain.iloc[:, -2])):
            df_test0 = remain.iloc[test0, :]
            tmp = data_folds[i]
            data_folds[i] = pd.concat([tmp, df_test0], axis=0)

    else:
        raise ValueError(os.path.basename(data_dir))

    # --------------------------------------------------------------

    for i in range(5):
        test_data = data_folds[i]
        print(test_data.shape)
        test_data_sc = my_scale(test_data)
        out0 = os.path.join(save_dir, f'index_data_label_lost_{i}_test.csv')
        if os.path.isfile(out0) and os.path.getsize(out0) > 0:
            t1 = test_data_sc.reset_index(drop=True).replace(np.nan, 9999)
            t2 = pd.read_csv(out0, header=None).replace(np.nan, 9999)
            t2.columns = t1.columns
            # print(abs(t1.values - t2.values).max())
            assert pd.testing.assert_frame_equal(t1, t2) is None
        else:
            test_data_sc.to_csv(out0, index=False, header=None)

        train_data = pd.concat([data_folds[i] for i in set(range(5)) - {i}], axis=0)
        train_data_sc = my_scale(train_data)
        out0 = os.path.join(save_dir, f'index_data_label_lost_{i}_train.csv')
        if os.path.isfile(out0) and os.path.getsize(out0) > 0:
            t1 = train_data_sc.reset_index(drop=True).replace(np.nan, 9999)
            t2 = pd.read_csv(out0, header=None).replace(np.nan, 9999)
            t2.columns = t1.columns
            # print(abs(t1.values - t2.values).max())
            assert pd.testing.assert_frame_equal(t1, t2) is None
        else:
            train_data_sc.to_csv(out0, index=False, header=None)

    alldata = pd.concat(data_folds, axis=0)
    dfnew = pd.merge(alldata, df['index'], on='index', how='outer')
    assert dfnew.shape[0] == df.shape[0]


def my_scale(data):
    data_sc = data.iloc[:, 1:-2].apply(lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)), axis=0)
    data_sc = pd.concat([data.iloc[:, 0], data_sc, data.iloc[:, -2:]], axis=1)

    return data_sc


def cut_data(df, lost, label):
    dfi = df[df.iloc[:, -1] == lost]
    # print(dfi.shape)
    if label == 2:
        dfii = dfi
    else:
        dfii = dfi[dfi.iloc[:, -2] == label]
    # print(dfii.shape)
    return dfii


def separate_data_cl2(df, lost, frac):
    df_sep = [cut_data(df, lost, i) for i in range(2)]
    df_1 = [dfi.iloc[:int(len(dfi) * frac), :] for dfi in df_sep]
    df_1 = pd.concat(df_1, axis=0)
    df_1.reset_index(drop=True, inplace=True)

    df_2 = [dfi.iloc[int(len(dfi) * frac):, :] for dfi in df_sep]
    df_2 = pd.concat(df_2, axis=0)
    df_2.reset_index(drop=True, inplace=True)

    return df_1, df_2


if __name__ == '__main__':
    main()
