from __future__ import print_function
import pandas as pd
import numpy as np
import argparse
import os
from joblib import Parallel, delayed
from parameters import rand_list_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./../data/')
    parser.add_argument('--data_name', type=str, default='nat1')
    parser.add_argument('--job', type=int, default=5, help='number of jobs for parallel')
    args = parser.parse_args()

    if '_' in args.data_name:
        drop_incomplete(args)
    elif args.data_name in ['nat1', 'nat2']:
        process_data_nat(args)
    elif args.data_name in ['full', 'MCAR', 'MAR', 'MNAR']:
        process_data_simu(args)
    elif args.data_name in ['uci']:
        process_data_uci(args)


def process_data_nat(args):
    rand_list = rand_list_dict[args.data_name]
    cmdGen = ' '.join([
        'python ./../preprocessing/data_generate_TADPOLE.py',
        '--data_root', args.data_root,
        '--data_name', args.data_name])
    os.system(cmdGen)
    cmdGen = ' '.join([
        'python ./../preprocessing/refine_miss.py',
        '--data_root', args.data_root,
        '--data_name', args.data_name])
    os.system(cmdGen)
    cmdRoot = ' '.join([
        'python ./../preprocessing/data_process_nat.py',
        '--data_root', args.data_root,
        '--data_name', args.data_name,
        '--rand '])
    Parallel(n_jobs=args.job)(delayed(os.system)(cmdRoot + str(k)) for k in rand_list)


def process_data_simu(args):
    full_rand = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14]
    if args.data_name in ['full']:
        cmdRoot = ' '.join([
            'python ./../preprocessing/data_process_simu.py',
            '--data_root', args.data_root,
            '--lost_num', '0'])
        Parallel(n_jobs=args.job)(delayed(os.system)(cmdRoot + f' --rand {rand}') for rand in full_rand)
    elif args.data_name in ['MCAR', 'MAR', 'MNAR']:
        cmdRoot = ' '.join([
            'python ./../preprocessing/data_process_simu.py',
            '--data_root', args.data_root,
            '--mode', args.data_name])
        if args.data_name == 'MCAR':
            Parallel(n_jobs=args.job)(delayed(os.system)(
                cmdRoot + f' --lost_num {lost_num} --p_inc {p_inc} --rand {rand}'
            ) for lost_num in [663] for p_inc in [0.05, 0.1, 0.15] for rand in rand_list_dict['MCAR'][f'L{lost_num}p{p_inc}'])
        elif args.data_name == 'MAR':
            Parallel(n_jobs=args.job)(delayed(os.system)(
                cmdRoot + f' --lost_num {lost_num} --p_inc {p_inc} --rand {rand}'
            ) for lost_num in [331] for p_inc in [0.2, 0.4, 0.6, 0.8] for rand in rand_list_dict['MAR'][f'L{lost_num}p{p_inc}'])
            Parallel(n_jobs=args.job)(delayed(os.system)(
                cmdRoot + f' --lost_num {lost_num} --p_inc {p_inc} --rand {rand}'
            ) for lost_num in [530] for p_inc in [0.5] for rand in rand_list_dict['MAR'][f'L{lost_num}p{p_inc}'])
            Parallel(n_jobs=args.job)(delayed(os.system)(
                cmdRoot + f' --lost_num {lost_num} --p_inc {p_inc} --rand {rand}'
            ) for lost_num in [596] for p_inc in [0.6] for rand in rand_list_dict['MAR'][f'L{lost_num}p{p_inc}'])
            Parallel(n_jobs=args.job)(delayed(os.system)(
                cmdRoot + f' --lost_num {lost_num} --p_inc {p_inc} --rand {rand}'
            ) for lost_num in [596] for p_inc in [0.8] for rand in rand_list_dict['MAR'][f'L{lost_num}p{p_inc}'])
        elif args.data_name == 'MNAR':
            Parallel(n_jobs=args.job)(delayed(os.system)(
                cmdRoot + f' --lost_num {lost_num} --p_inc {p_inc} --rand {rand}'
            ) for lost_num in [663] for p_inc in [0.05, 0.1, 0.15] for rand in rand_list_dict['MNAR'][f'L{lost_num}p{p_inc}'])


def drop_incomplete(args):
    data_name, drop = args.data_name.split('_')
    rand_list = rand_list_dict[data_name]
    cmdRoot = ' '.join([
        'python ./../preprocessing/drop_incomplete.py',
        '--data_root', args.data_root,
        '--data_name', data_name,
        '--drop', drop])
    Parallel(n_jobs=args.job)(delayed(os.system)(cmdRoot + f' --rand {rand}') for rand in rand_list)
    # for AutoMetric & GCNRisk require 4 features: age, gender, education, APOE
    Parallel(n_jobs=args.job)(delayed(os.system)(cmdRoot + f' --rand {rand} -ex4') for rand in rand_list)


def process_data_uci(args):
    cmdGen = ' '.join([
        'python ./../preprocessing/data_generate_UCI.py',
        '--data_root', args.data_root])
    Parallel(n_jobs=args.job)(delayed(os.system)(
        cmdGen + f' --data_name {data_name}'
    ) for data_name in ['BC', 'CC', 'CK', 'HC', 'HD', 'HP', 'HS', 'PI'])
    cmdRoot = ' '.join([
        'python ./../preprocessing/data_process_uci.py',
        '--data_root', args.data_root])
    Parallel(n_jobs=args.job)(delayed(os.system)(
        cmdRoot + f' --data_name {data_name} --rand {rand}'
    ) for data_name in ['BC', 'CC', 'CK', 'HC', 'HD', 'HP', 'HS', 'PI'] for rand in rand_list_dict[data_name])


if __name__ == '__main__':
    main()
