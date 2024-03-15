"""
Batch run RF & SVM for all datasets, combined with deletion or imputation
"""
import numpy as np
import pandas as pd
import argparse
import os
import time
from joblib import Parallel, delayed
from itertools import product, permutations
from parameters import rand_list_dict, seed_RF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./../results/')
    parser.add_argument('--data_root', type=str, default='./../data/')
    parser.add_argument('--data_name', type=str, default='nat1')
    parser.add_argument('--lost_name', type=str, default='L999', help='number of missing features')
    parser.add_argument('--classify', type=str, default='', choices=['RF', 'SVM'])
    parser.add_argument('--job', type=int, default=5, help='number of jobs for parallel')
    args = parser.parse_args()

    if args.data_name in ['nat1', 'nat2', 'BC', 'CC', 'CK', 'HC', 'HD', 'HP', 'HS', 'PI']:
        assert args.lost_name == 'L999'
        classify(args)
    elif args.data_name in ['MCAR', 'MAR', 'MNAR']:
        assert args.lost_name != 'L999'
        classify_simu(args)
    elif args.data_name in ['nat1_dropFeature', 'nat1_dropSample', 'nat2_dropFeature']:
        assert args.lost_name == 'L999'
        classify_drop(args)
    else:
        raise ValueError(f'Invalid data_name: {args.data_name}')


def classify(args):
    fill_methods = ['mean', 'softimpute', 'knn.3']  # , 'missForest'
    rand_list = rand_list_dict[args.data_name]

    cmdRoot = ' '.join([
        'python ./../comparing/RFSVM/main.py',
        '--save_dir', args.save_dir,
        '--data_root', args.data_root,
        '--data_name', args.data_name,
        '--classify', args.classify])

    cmd_list = []
    have, total = 0, 0
    for fill in fill_methods:
        seed_list = RF_seed[args.data_name][f'{fill}+RF']
        for (rand, seed), fold in product(zip(rand_list, seed_list), range(5)):
            total += 1
            if not fail_flag(os.path.join(args.save_dir, f'Impute{args.classify}'),
                             args.data_name, fill, rand, fold):
                have += 1
                continue
            cmdParams = ' --rand %d --fold %d --fill %s --seed %d' % (rand, fold, fill, seed)
            cmd_list.append(' '.join([cmdRoot, cmdParams]))

    print(len(cmd_list), have, total)
    Parallel(n_jobs=args.job)(delayed(os.system)(cmd) for cmd in cmd_list)


def classify_simu(args):
    fill_methods = ['mean', 'softimpute', 'knn.3']  # , 'missForest'
    rand_list = rand_list_dict[args.data_name][args.lost_name]

    cmdRoot = ' '.join([
        'python ./../comparing/RFSVM/main.py',
        '--save_dir', args.save_dir,
        '--data_root', args.data_root,
        '--data_name', args.data_name,
        '--classify', args.classify])

    cmd_list = []
    have, total = 0, 0
    for fill in fill_methods:
        seed_list = RF_seed[args.data_name][args.lost_name][f'{fill}+RF']
        for (rand, seed), fold in product(zip(rand_list, seed_list), range(5)):
            total += 1
            if not fail_flag(os.path.join(args.save_dir, f'Impute{args.classify}'),
                             args.data_name, fill, rand, fold, args.lost_name):
                have += 1
                continue
            cmdParams = ' --lost_name %s --rand %d --fold %d --fill %s --seed %d' % (args.lost_name, rand, fold, fill, seed)
            cmd_list.append(' '.join([cmdRoot, cmdParams]))

    print(len(cmd_list), have, total)
    Parallel(n_jobs=args.job)(delayed(os.system)(cmd) for cmd in cmd_list)


def classify_drop(args):
    data_name, drop = args.data_name.split('_drop')
    drop = 'DA' if drop == 'Feature' else 'DS'
    rand_list = rand_list_dict[data_name]

    cmdRoot = ' '.join([
        'python ./../comparing/RFSVM/main.py',
        '--save_dir', args.save_dir,
        '--data_root', args.data_root,
        '--data_name', args.data_name,
        '--classify', args.classify])

    cmd_list = []
    have, total = 0, 0
    seed_list = RF_seed[data_name][f'{drop}+RF']
    for (rand, seed), fold in product(zip(rand_list, seed_list), range(5)):
        total += 1
        if not fail_flag(os.path.join(args.save_dir, f'Impute{args.classify}'),
                         args.data_name, '', rand, fold, args.lost_name):
            have += 1
            continue
        cmdParams = ' --rand %d --fold %d --seed %d ' % (rand, fold, seed)
        cmd_list.append(' '.join([cmdRoot, cmdParams]))

    print(len(cmd_list), have, total)
    Parallel(n_jobs=args.job)(delayed(os.system)(cmd) for cmd in cmd_list)


def fail_flag(save_path, data_name, fill, rand, fold, lost_name='L999', seed=2021):
    if (data_name.startswith('nat') and 'drop' in data_name) or data_name == 'full':
        assert fill == '' and lost_name == 'L999'
        data_dir = os.path.join(save_path, data_name, f'rand{rand}', lost_name, f's{seed}', f'fold{fold}')
    else:
        data_dir = os.path.join(save_path, data_name, fill, f'rand{rand}', lost_name, f's{seed}', f'fold{fold}')
    if os.path.exists(data_dir) and set(os.listdir(data_dir)) == \
            {'test_indicators.csv', 'train_indicators.csv', 'test_label.prob.pred.csv', 'train_label.prob.pred.csv'}:
        return False
    else:
        return True


if __name__ == '__main__':
    main()
