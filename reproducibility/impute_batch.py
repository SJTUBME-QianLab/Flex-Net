"""
Batch run imputation
"""
import numpy as np
import pandas as pd
import argparse
import os
import time
from joblib import Parallel, delayed
from itertools import product, permutations
from parameters import rand_list_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./../data/imputed_data/')
    parser.add_argument('--data_root', type=str, default='./../data/')
    parser.add_argument('--data_name', type=str, default='nat1')
    parser.add_argument('--lost_name', type=str, default='L999', help='number of missing features')
    parser.add_argument('--job', type=int, default=5, help='number of jobs for parallel')
    args = parser.parse_args()

    if args.data_name in ['nat1', 'nat2', 'BC', 'CC', 'CK', 'HC', 'HD', 'HP', 'HS', 'PI']:
        assert args.lost_name == 'L999'
        impute(args)
    elif args.data_name in ['MCAR', 'MAR', 'MNAR']:
        assert args.lost_name != 'L999'
        impute_simu(args)
    else:
        raise ValueError(f'Invalid data_name: {args.data_name}')


def impute(args):
    fill_methods = ['mean', 'softimpute', 'knn.3']  # , 'missForest'
    rand_list = rand_list_dict[args.data_name]

    cmdRoot = ' '.join([
        'python ./../comparing/impute/impute.py',
        '--save_dir', args.save_dir,
        '--data_root', args.data_root,
        '--data_name', args.data_name])
    cmdRoot1 = ' '.join([
        'R --no-save <./../comparing/impute/missForest.R',
        args.data_name])
    cmd_list = []
    have, total = 0, 0
    for rand, fold, fill in product(rand_list, range(5), fill_methods):
        total += 1
        if not fail_flag(args.save_dir, args.data_name, fill, rand, fold):
            have += 1
            continue
        if fill == 'missForest':
            cmdParams = ' %s %d %d ' % ('L999', rand, fold)
            cmd_list.append(' '.join([cmdRoot1, cmdParams]))
        else:
            cmdParams = ' --rand %d --fold %d --fill %s ' % (rand, fold, fill)
            cmd_list.append(' '.join([cmdRoot, cmdParams]))
        # print(rand, fold, fill)

    print(len(cmd_list), have, total)
    Parallel(n_jobs=args.job)(delayed(os.system)(cmd) for cmd in cmd_list)


def impute_simu(args):
    fill_methods = ['mean', 'softimpute', 'knn.3']  # , 'missForest'
    rand_list = rand_list_dict[args.data_name][args.lost_name]

    cmdRoot = ' '.join([
        'python ./../comparing/impute/impute.py',
        '--save_dir', args.save_dir,
        '--data_root', args.data_root,
        '--data_name', args.data_name])
    cmdRoot1 = ' '.join([
        'R --no-save <./../comparing/impute/missForest.R',
        args.data_name])
    cmd_list = []
    have, total = 0, 0
    for rand, fold, fill in product(rand_list, range(5), fill_methods):
        total += 1
        if not fail_flag(args.save_dir, args.data_name, fill, rand, fold, args.lost_name):
            have += 1
            continue
        if fill == 'missForest':
            cmdParams = ' %s %d %d ' % (args.lost_name, rand, fold)
            cmd_list.append(' '.join([cmdRoot1, cmdParams]))
        else:
            cmdParams = ' --lost_name %s --rand %d --fold %d --fill %s ' % (args.lost_name, rand, fold, fill)
            cmd_list.append(' '.join([cmdRoot, cmdParams]))
        # print(rand, fold, fill)

    print(len(cmd_list), have, total)
    Parallel(n_jobs=args.job)(delayed(os.system)(cmd) for cmd in cmd_list)


def fail_flag(save_path, data_name, fill, rand, fold, lost_name='L999'):
    data_dir = os.path.join(save_path, data_name, fill, f'rand{rand}', lost_name)
    if os.path.exists(os.path.join(data_dir, f'train{fold}.csv')) and \
            os.path.exists(os.path.join(data_dir, f'test{fold}.csv')):
        return False
    else:
        # print(data_dir)
        return True


if __name__ == '__main__':
    main()
