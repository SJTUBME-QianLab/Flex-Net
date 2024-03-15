import numpy as np
import pandas as pd
import os
import time
from joblib import Parallel, delayed
from itertools import product, permutations
data_root = '/home/data/tangxl/MetaMissing/data/ADreal/'
save_dir = '/home/data/tangxl/MetaMissing/results/20230706-AutoMetric/'
name_dict = {
    'Nat14dropSample': '14_dropSample_agep+cog9+mriL+mriX_',
    'Nat14dropFeature': '14_dropFeature_agep+cog9+mriL+mriX_',
    'Nat12dropFeature': '12_dropFeature_agep+mriL+mriX+csf3+petms3_',
}


def main():

    # csv2pkl.py
    # -------------

    # rand_list = [1, 2, 6, 8, 9, 10, 12, 14, 16, 17]
    # # # rand_list = list(range(1, 18))
    # # mainParams('Nat14dropSample', rand_list)
    # # mainParams('Nat14dropFeature', rand_list)
    # mainParams('Nat14dropSample', rand_list, [2019, 42, 4377, 666])

    rand_list = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
    mainParams('Nat12dropFeature', rand_list)


def mainParams(data_name, rand_list, seed_list=None):
    if seed_list is None:
        seed_list = [2019]
    N_way, N_shots, batch_size, dec_lr, lr = 3, 10, 64, 300, 0.01

    cmdRoot = 'python main_AD_MCI_NL.py --data_root %s --save_dir %s ' % \
              (data_root+name_dict[data_name], save_dir+data_name)
    if data_name == 'Nat14dropFeature':
        cmdRoot += '--feature_num 638 --w_feature_num 634 '
    if data_name == 'Nat12dropFeature':
        cmdRoot += '--feature_num 630 --w_feature_num 626 '
    cmd_list = []
    have, total = 0, 0
    for rand, fold, seed in product(rand_list, range(5), seed_list):
        exp_name = 'N{:d}s{:d}b{:d}dec{:d}lr{:g}it{:d}se{:d}__'.format(
            N_way, N_shots, batch_size, dec_lr, lr, 500, seed)
        total += 1
        if not fail_flag(save_dir+data_name, rand, fold, exp_name, rm=True):
            have += 1
            continue
        cmdParams = ' --test_N_way %d --train_N_way %d --test_N_shots %d --train_N_shots %d --batch_size %d ' \
            '--dec_lr %d --lr %s --random_seed %d' \
            % (N_way, N_way, N_shots, N_shots, batch_size, dec_lr, str(lr), seed)
        cmdFold = ' --rand %d --fold %d ' % (rand, fold)
        cmd_list.append(''.join([cmdRoot, cmdParams, cmdFold]))

    print(len(cmd_list), have, total)
    half = int(len(cmd_list) * 0.5)
    cmd_list0 = ['CUDA_VISIBLE_DEVICES=0 ' + x for x in cmd_list[:half]]
    cmd_list1 = ['CUDA_VISIBLE_DEVICES=1 ' + x for x in cmd_list[half:]]
    Parallel(n_jobs=2)(delayed(device1)(ll, 2) for ll in [cmd_list0, cmd_list1])

    # for cmd in cmd_list:
    #     os.system(cmd)

    # Parallel(n_jobs=2)(delayed(os.system)(cmd) for cmd in cmd_list)


def rename():
    for rand in os.listdir(save_dir):
        randNum = int(rand.split('rand')[-1])
        for L in os.listdir(os.path.join(save_dir, rand)):
            for i in range(5):
                path0 = os.path.join(save_dir, rand, L, f'fold{i}')
                if not os.path.exists(path0):
                    print(path0)
                    continue
                for ff in os.listdir(path0):
                    # new_name = '__'.join(ff.split('__')[:-1]) + f's{randNum}__'
                    # # print(ff, new_name)
                    # os.rename(os.path.join(path0, ff), os.path.join(path0, new_name))
                    new_name = '__'.join(ff.split(f's{randNum}__')[:-1]) + f'__'
                    # print(ff, new_name)
                    os.rename(os.path.join(path0, ff), os.path.join(path0, new_name))


def fail_flag(path, rand, fold, exp_name, rm=False):
    dir_i = os.path.join(path, f"rand{rand}", f"fold{fold}")
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

