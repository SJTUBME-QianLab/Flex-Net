"""
Batch run GCNRisk for nat1-DA, nat1-DS, nat2-DS
"""
import numpy as np
import pandas as pd
import argparse
import os
import time
from joblib import Parallel, delayed
from itertools import product, permutations
from parameters import rand_list_dict, get_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./../results/')
    parser.add_argument('--data_root', type=str, default='./../data/')
    parser.add_argument('--data_name', type=str, default='nat1')
    parser.add_argument('--job', type=int, default=5, help='number of jobs for parallel')
    args = parser.parse_args()

    assert args.data_name in ['nat1_dropFeature', 'nat1_dropSample', 'nat2_dropFeature']
    GCNrisk_drop(args)


def GCNrisk_drop(args):
    data_name, drop = args.data_name.split('_drop')
    drop = 'DA' if drop == 'Feature' else 'DS'
    rand_list = rand_list_dict[data_name]
    para_name0 = 'cheby.ori.3.dr0.1dec1e-05h64lr0.01e400'

    cmdRoot = ' '.join([
        'python ./../comparing/GCN/main.py',
        '--save_dir', args.save_dir,
        '--data_root', args.data_root,
        '--data_name', args.data_name])
    cmd_list = []
    have, total = 0, 0
    seed_list = get_seed(f'{drop}+GCN+risk')[data_name]
    for (rand, seed), fold in product(zip(rand_list, seed_list), range(5)):
        exp_name = f'{para_name0}s{seed}__'
        total += 1
        if not fail_flag(os.path.join(args.save_dir, 'GCNrisk'),
                         args.data_name, '', rand, fold, exp_name, rm=False):
            have += 1
            continue
        cmdFold = ' --rand %d --fold %d --seed %d ' % (rand, fold, seed)
        cmd_list.append(' '.join([cmdRoot, cmdFold]))

    # cmd_list = cmd_list[:2]
    print(len(cmd_list), have, total)
    # half = int(len(cmd_list) * 0.5)
    # cmd_list0 = ['CUDA_VISIBLE_DEVICES=0 ' + x for x in cmd_list[:half]]
    # cmd_list1 = ['CUDA_VISIBLE_DEVICES=1 ' + x for x in cmd_list[half:]]
    # Parallel(n_jobs=2)(delayed(device1)(ll, args.job) for ll in [cmd_list0, cmd_list1])


def fail_flag(save_path, data_name, fill, rand, fold, exp_name='', lost_name='L999', rm=False):
    if (data_name.startswith('nat') and 'drop' in data_name) or data_name == 'full':
        assert fill == '' and lost_name == 'L999'
        dir_i = os.path.join(save_path, data_name, f'rand{rand}', lost_name, f'fold{fold}')
    else:
        dir_i = os.path.join(save_path, data_name, fill, f'rand{rand}', lost_name, f'fold{fold}')
    flag = False
    if not os.path.isdir(dir_i):
        flag = True
        # print('not exist,', dir_i)
    else:
        # file = os.listdir(dir_i)
        # assert len(file) == 1
        file = [kk for kk in sorted(os.listdir(dir_i)) if exp_name == '__'.join(kk.split('__')[1:])]
        if len(file) == 0:
            flag = True
            # print('no file,', dir_i)
        else:
            for ff in file:
                if not os.path.isfile(os.path.join(dir_i, ff, 'test_indicators.csv')):
                    flag = True
                    print(os.path.join(dir_i, ff))
                    if rm:
                        os.system('rm -R {}'.format(os.path.join(dir_i, ff)))

    # extra
    if not flag:
        file = [kk for kk in sorted(os.listdir(dir_i)) if exp_name == '__'.join(kk.split('__')[1:])]
        for ff in file[1:]:
            assert os.path.isfile(os.path.join(dir_i, ff, 'test_indicators.csv'))
            print('extra ' + os.path.join(dir_i, ff))
            if rm:
                os.system('rm -R {}'.format(os.path.join(dir_i, ff)))
    return flag


def device1(cmd_list, job=1):
    Parallel(n_jobs=job)(delayed(os.system)(cmd) for cmd in cmd_list)


if __name__ == '__main__':
    main()
