"""
Due to virtual environment conflicts, 
we have to conduct imputation and classification separately.
This code is used to obtain imputed data based on different imputation algorithms
"""

from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
import random

import fancyimpute
import miceforest


def main():
    # Training settings 训练参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./../../data/imputed_data/')
    parser.add_argument('--data_root', type=str, default='./../../data/')
    parser.add_argument('--data_name', type=str, default='nat1')
    parser.add_argument('--rand', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--fold', type=int, default=0, help='fold index, [0,1,2,3,4]')
    parser.add_argument('--fill', type=str, default='mean', help='method for fill NaN')
    parser.add_argument('--lost_name', type=str, default='L999', help='number of missing features')

    args = parser.parse_args()

    seed = 2021
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    output = os.path.join(args.save_dir, args.data_name, args.fill, f'rand{args.rand}', args.lost_name)
    os.makedirs(output, exist_ok=True)

    train_x, test_x = load_data(args)
    train_x, test_x = drop_allna(train_x, test_x)
    train_x_imputed = pd.concat(
        [train_x.iloc[:, 0], pd.DataFrame(Impute(train_x.iloc[:, 1:-2], args.fill)), train_x.iloc[:, -2:]],
        axis=1)
    test_x_imputed = pd.concat(
        [test_x.iloc[:, 0], pd.DataFrame(Impute(test_x.iloc[:, 1:-2], args.fill)), test_x.iloc[:, -2:]],
        axis=1)

    pd.DataFrame(train_x_imputed).to_csv(os.path.join(output, f'train{args.fold}.csv'), index=False, header=None)
    pd.DataFrame(test_x_imputed).to_csv(os.path.join(output, f'test{args.fold}.csv'), index=False, header=None)


def Impute(df, fill):
    fill = fill.split('.')
    if fill[0] == 'mean':
        df_impute = fancyimpute.SimpleFill(fill_method="mean").fit_transform(df)
    elif fill[0] == 'knn':
        df_impute = fancyimpute.KNN(k=int(fill[1])).fit_transform(df)
    elif fill[0] == 'softimpute':
        df_impute = fancyimpute.SoftImpute().fit_transform(df.values)
    elif fill[0] == 'miceforest':
        kds = miceforest.KernelDataSet(df, save_all_iterations=True, random_state=args.seed)
        kds.mice(int(fill[1]))  # 4
        df_impute = kds.complete_data()
    else:
        raise ValueError(args.fill)

    return df_impute


def drop_allna(train, test):
    tr_nan = np.where(np.isnan(train).sum(axis=0) == train.shape[0])[0].tolist()
    te_nan = np.where(np.isnan(test).sum(axis=0) == test.shape[0])[0].tolist()
    if len(tr_nan) == 0:
        idx = te_nan
    elif len(te_nan) == 0:
        idx = tr_nan
    else:
        idx = tr_nan + te_nan
    tr = train.drop(train.columns[idx], axis=1)
    te = test.drop(train.columns[idx], axis=1)
    return tr, te


def load_data(args):
    if args.data_name in ['BC', 'CC', 'CK', 'HC', 'HD', 'HP', 'HS', 'PI']:
        assert args.lost_name == 'L999'
        data_dir = os.path.join(args.data_root, 'uci', args.data_name, f'divide_{args.rand}')
        data_tr = pd.read_csv(os.path.join(data_dir, f'index_data_label_lost_{args.fold}_train.csv'), header=None)
        data_te = pd.read_csv(os.path.join(data_dir, f'index_data_label_lost_{args.fold}_test.csv'), header=None)
    elif args.data_name in ['MCAR', 'MAR', 'MNAR']:
        assert args.lost_name != 'L999'
        data_dir = os.path.join(args.data_root, 'simu', f'simu_1100_{args.data_name}', f'divide_{args.rand}', args.lost_name)
        data_tr = pd.read_csv(os.path.join(data_dir, f'index_data_label_lost_{args.fold}_train.csv'), header=None)
        data_te = pd.read_csv(os.path.join(data_dir, f'index_data_label_lost_{args.fold}_test.csv'), header=None)
    elif args.data_name in ['nat1', 'nat2']:
        assert args.lost_name == 'L999'
        data_dir = os.path.join(args.data_root, 'nat', f'{args.data_name}_0.8_', f'divide_{args.rand}')
        data_tr = pd.read_csv(os.path.join(data_dir, f'index_data_label_lost_{args.fold}_train.csv'), header=None)
        data_te = pd.read_csv(os.path.join(data_dir, f'index_data_label_lost_{args.fold}_test.csv'), header=None)
    else:
        raise ValueError(args.data_name)
    # ------------------------------------------

    return data_tr, data_te


if __name__ == '__main__':
    main()
