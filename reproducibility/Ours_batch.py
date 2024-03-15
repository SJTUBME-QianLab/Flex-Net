import pandas as pd
import numpy as np
import argparse
import os
import time
from joblib import Parallel, delayed
from itertools import product, permutations
from parameters import rand_list_dict, params_ours, get_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./../results/')
    parser.add_argument('--data_root', type=str, default='./../data/')
    parser.add_argument('--data_name', type=str, default='nat1')
    parser.add_argument('--lost_name', type=str, default='L999', help='number of missing features')
    parser.add_argument('--job', type=int, default=2, help='number of jobs for parallel')
    args = parser.parse_args()

    if args.data_name in ['nat1', 'nat2', 'BC', 'CC', 'CK', 'HC', 'HD', 'HP', 'HS', 'PI']:
        assert args.lost_name == 'L999'
        ours(args)
    elif args.data_name in ['MCAR', 'MAR', 'MNAR']:
        assert args.lost_name != 'L999'
        ours_simu(args)
    else:
        raise ValueError(f'Invalid data_name: {args.data_name}')


def ours(args):
    rand_list = rand_list_dict[args.data_name]
    seed_list = get_seed('Ours')[args.data_name]
    hparams = dict()
    for k, v in params_ours[args.data_name].items():
        if k == 'N_way':
            hparams['test_N_way'] = v
            hparams['train_N_way'] = v
        elif k == 'batch_size':
            hparams['batch_size'] = v
            hparams['batch_size_test'] = v
        elif k == 'para_name0':
            para_name0 = v
        else:
            hparams[k] = v
    cmdParams = ' '.join([f'--{k} {v}' for k, v in hparams.items()])

    main = 'main_nat.py' if args.data_name in ['nat1', 'nat2'] else 'main_uci.py'
    cmdRoot = ' '.join([
        f'python ./../Flex-Net/{main}',
        '--save_dir', args.save_dir,
        '--data_root', args.data_root,
        '--data_name', args.data_name])
    cmd_list = []
    have, total = 0, 0
    for (rand, seed), fold in product(zip(rand_list, seed_list), range(5)):
        exp_name = f'{para_name0}s{seed}__'
        total += 1
        dir_i = os.path.join(args.save_dir, args.data_name, f"rand{rand}", f"fold{fold}")
        if not fail_flag(dir_i, exp_name, rm=False):  # not failed, skip
            # print(dd['data_name'], exp_name, ' exists.')
            have += 1
            continue
        cmdFold = ' --rand %d --fold %d --seed %d ' % (rand, fold, seed)
        cmd_list.append(' '.join([cmdRoot, cmdParams, cmdFold]))

    print(len(cmd_list), have, total)
    half = int(len(cmd_list) * 0.5)
    cmd_list0 = ['CUDA_VISIBLE_DEVICES=0 ' + x for x in cmd_list[:half]]
    cmd_list1 = ['CUDA_VISIBLE_DEVICES=1 ' + x for x in cmd_list[half:]]
    Parallel(n_jobs=2)(delayed(device1)(ll, args.job) for ll in [cmd_list0, cmd_list1])


def ours_simu(args):
    rand_list = rand_list_dict[args.data_name][args.lost_name]
    seed_list = get_seed('Ours')[args.data_name][args.lost_name]
    hparams = dict()
    for k, v in params_ours[args.data_name].items():
        if k == 'N_way':
            hparams['test_N_way'] = v
            hparams['train_N_way'] = v
        elif k == 'batch_size':
            hparams['batch_size'] = v
            hparams['batch_size_test'] = v
        elif k == 'para_name0':
            para_name0 = v
        else:
            hparams[k] = v
    cmdParams = ' '.join([f'--{k} {v}' for k, v in hparams.items()])
    lost_num, p_inc = args.lost_name.split('L')[1].split('p')
    cmdParams += f' --lost_num {lost_num} --p_inc {p_inc}'

    cmdRoot = ' '.join([
        'python ./../Flex-Net/main_simu.py',
        '--save_dir', args.save_dir,
        '--data_root', args.data_root,
        '--data_name', args.data_name])
    cmd_list = []
    have, total = 0, 0
    for (rand, seed), fold in product(zip(rand_list, seed_list), range(5)):
        exp_name = f'{para_name0}s{seed}__'
        total += 1
        dir_i = os.path.join(args.save_dir, args.data_name, f"rand{rand}", f"fold{fold}")
        if not fail_flag(dir_i, exp_name, rm=False):  # not failed, skip
            # print(dd['data_name'], exp_name, ' exists.')
            have += 1
            continue
        cmdFold = ' --rand %d --fold %d --seed %d ' % (rand, fold, seed)
        cmd_list.append(' '.join([cmdRoot, cmdParams, cmdFold]))

    print(len(cmd_list), have, total)
    half = int(len(cmd_list) * 0.5)
    cmd_list0 = ['CUDA_VISIBLE_DEVICES=0 ' + x for x in cmd_list[:half]]
    cmd_list1 = ['CUDA_VISIBLE_DEVICES=1 ' + x for x in cmd_list[half:]]
    Parallel(n_jobs=2)(delayed(device1)(ll, args.job) for ll in [cmd_list0, cmd_list1])


def fail_flag(dir_i, exp_name, rm=False):
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
                if not (os.path.isfile(os.path.join(dir_i, ff, 'test_indicators.csv')) and
                        os.path.isfile(os.path.join(dir_i, ff, 'test_predictions.csv'))):
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
