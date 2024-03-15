import pandas as pd
import numpy as np
import argparse
import os
import pickle
from sklearn.model_selection import StratifiedKFold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./../data/')
    parser.add_argument('--data_name', type=str, default='nat1')
    parser.add_argument('--drop', type=str, choices=['Feature', 'Sample'])
    parser.add_argument('--rand', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('-ex4', action="store_true")
    args = parser.parse_args()
    print(args.ex4)

    if args.data_name == 'nat2':
        assert args.drop != 'Sample'
    data_name0 = args.data_name + '_0.8_'
    data_name = data_name0.replace('0.8', f'drop{args.drop}')
    data_path0 = os.path.join(args.data_root, 'nat', data_name0).rstrip('/')
    data_path = os.path.join(args.data_root, 'nat', data_name).rstrip('/')
    if not os.path.isfile(data_path + 'value.csv'):
        pre(data_path0, data_name)
    divide(data_path, data_name0, args.rand, ex4=args.ex4)
    if not args.ex4:
        csv2pkl(data_path, args.rand)


def pre(data_root, data_name):
    data_raw = pd.read_csv(os.path.join(data_root, 'index_data_label_lost.csv'), header=None)
    if data_name.startswith('nat1'):
        if 'dropSample' in data_name:
            data_full = data_raw[data_raw.iloc[:, -1] == 0]
            print(data_full.shape)
            assert data_full.shape[0] == 3499 - 2437 and data_full.shape[1] == 669 + 3
            df = data_raw.dropna(axis=0)
            assert df.shape == data_full.shape
        elif 'dropFeature' in data_name:
            data_full = data_raw.dropna(axis=1)
            print(data_full.shape)
            assert data_full.shape[0] == 3499 and data_full.shape[1] == 669 + 3 - 31
        else:
            raise ValueError(data_name)
    elif data_name.startswith('nat2'):
        if 'dropFeature' in data_name:
            data_full = data_raw.dropna(axis=1)
            print(data_full.shape)
            assert data_full.shape[0] == 3597 and data_full.shape[1] == 630 + 3
        else:
            raise ValueError(data_name)
    else:
        raise ValueError(data_name)
    data_full = data_full.iloc[:, :-1]
    data_full.to_csv(os.path.join(data_root, '..', f'{data_name}value.csv'), index=False, header=None)


def divide(data_dir, data_name0, rand, ex4=False):
    if ex4:
        save_dir = os.path.join(data_dir+'with4risk', f'divide_{rand}')
    else:
        save_dir = os.path.join(data_dir, f'divide_{rand}')
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(data_dir + 'value.csv', header=None)

    if 'dropSample' in data_dir:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rand)
        data_folds = []
        for [_, test0] in list(skf.split(range(len(df)), df.iloc[:, -1])):
            df_test0 = df.iloc[test0, :]
            data_folds.append(df_test0)
    elif 'dropFeature' in data_dir:
        raw = os.path.join(data_dir, '..', data_name0, f'divide_{rand}')
        data_folds = []
        for i in range(5):
            raw_test = pd.read_csv(os.path.join(raw, f'index_data_label_lost_{i}_test.csv'), header=None)
            raw_test = raw_test.iloc[:, [0, raw_test.shape[1] - 2]]
            # print(raw_test.columns, df.columns[:5])
            df_test0 = pd.merge(raw_test.iloc[:, [0]], df, how='left')
            # print(df_test0.columns[:5])
            # print(df_test0.iloc[:5, -1].values, raw_test.iloc[:5, -1].values)
            assert (df_test0.iloc[:, -1].values == raw_test.iloc[:, -1].values).all()
            assert (df_test0.iloc[:, 0] == raw_test.iloc[:, 0]).all()
            data_folds.append(df_test0)
    else:
        raise ValueError(data_dir)

    for i in range(5):
        outTe = os.path.join(save_dir, f'index_data_label_{i}_test.csv')
        outTr = os.path.join(save_dir, f'index_data_label_{i}_train.csv')
        if not (os.path.isfile(outTe) and os.path.getsize(outTe) > 0) or i == 4:
            test_data = data_folds[i]
            test_data_sc = my_scale(test_data, ex4)
            train_data = pd.concat([data_folds[i] for i in set(range(5)) - {i}], axis=0)
            train_data_sc = my_scale(train_data, ex4)
        if not (os.path.isfile(outTe) and os.path.getsize(outTe) > 0):
            test_data_sc.to_csv(outTe, index=False, header=None)
            train_data_sc.to_csv(outTr, index=False, header=None)
        if (os.path.isfile(outTe) and os.path.getsize(outTe) > 0) and i == 4:
            t1 = test_data_sc.reset_index(drop=True)
            t2 = pd.read_csv(outTe, header=None)
            t1.columns = t2.columns
            pd.testing.assert_frame_equal(t1, t2)
            t1 = train_data_sc.reset_index(drop=True)
            t2 = pd.read_csv(outTr, header=None)
            t1.columns = t2.columns
            pd.testing.assert_frame_equal(t1, t2)

        print('finish divide-scale-save data ' + str(i))

    alldata = pd.concat(data_folds, axis=0)
    dfnew = pd.merge(alldata, df[0], on=0, how='outer')
    assert dfnew.shape[0] == df.shape[0]


def my_scale(data, ex4=False):
    data_sc = data.iloc[:, 1:-1].apply(lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)), axis=0)
    if ex4:
        data_sc = pd.concat([data.iloc[:, :5], data_sc, data.iloc[:, -1:]], axis=1)  # [0]+[1~4]agep
    else:
        data_sc = pd.concat([data.iloc[:, 0], data_sc, data.iloc[:, -1:]], axis=1)  # [0]
    return data_sc


def csv2pkl(data_dir, rand):
    for fold in range(5):
        for partition in ['train', 'test']:
            out_file = os.path.join(data_dir, f'divide_{rand}', f'{partition}{fold}.pkl')
            if not (os.path.isfile(out_file) and os.path.getsize(out_file) > 0) or fold == 4:
                df = pd.read_csv(os.path.join(data_dir, f'divide_{rand}', f'index_data_label_{fold}_{partition}.csv'), header=None)
                df_new = change_format(df)
            if not (os.path.isfile(out_file) and os.path.getsize(out_file) > 0):
                with open(out_file, 'wb') as f:
                    pickle.dump(df_new, f)
                print(f'save {rand}, {fold}, {partition}')
            if (os.path.isfile(out_file) and os.path.getsize(out_file) > 0) and fold == 4:
                with open(out_file, 'rb') as f:
                    exist = pickle.load(f)
                for kk in ['CN', 'MCI', 'AD']:
                    print(np.vstack(exist[kk]).shape)
                    assert (np.vstack(exist[kk]) == np.vstack(df_new[kk])).all()


def change_format(data):
    label_dict = {0: 'CN', 1: 'MCI', 2: 'AD'}
    data_new = {}
    for i in range(3):
        ddi = data[data.iloc[:, -1] == i]
        ddi = ddi.iloc[:, 1:-1].values
        data_new[label_dict[i]] = []
        for nn in range(len(ddi)):
            data_new[label_dict[i]].append([ddi[nn, :]])
    return data_new


if __name__ == '__main__':
    main()
