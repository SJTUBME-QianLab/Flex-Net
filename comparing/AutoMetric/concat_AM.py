import pandas as pd
import numpy as np
import os
import time
import pickle
import re
from tools.utils import eval_metric, get_auc
from joblib import Parallel, delayed
out_path = '/home/tangxl/MetaMissing/202306-evaluation/ADratio'
root_path = '/home/data/tangxl/MetaMissing/results/'
metrics = ['acc', 'pre', 'sen', 'spe', 'f1', 'auc']
ave = 'macro'


def main():
    lost_nums = [33, 66, 99]
    fill_methods = ['mean', 'softimpute', 'knn.3', 'missForest', 'random']  # miceforest.4

    # # CorrectMetrics(1, 33, 0)
    # Parallel(n_jobs=5)(delayed(CorrectMetrics)(rand, lost, fold) for rand in range(1, 13)
    #                    for lost in lost_nums for fold in range(5))

    # # ours
    # Parallel(n_jobs=5)(delayed(concatRandOurs)(lost) for lost in lost_nums)

    # # impute+RF/SVM
    # # concatRandRFSVM(33, 'mean', clf='RF')
    # Parallel(n_jobs=5)(delayed(concatRandRFSVM)(lost, imp, clf='RF') for lost in lost_nums
    #                    for imp in fill_methods)
    # Parallel(n_jobs=5)(delayed(concatRandRFSVM)(lost, imp, clf='SVM') for lost in lost_nums
    #                    for imp in fill_methods)

    # # impute+GCN
    # # concatRandGCN(33, 'mean')
    # Parallel(n_jobs=5)(delayed(concatRandGCN)(lost, imp) for lost in lost_nums[:1]
    #                    for imp in fill_methods[:3])

    # ##############################################
    # total
    # methods_list = [f'one_{kk}+{tt}' for kk in fill_methods for tt in ['RF', 'SVM']]
    # methods_list += [f'accMax_{kk}+GCN' for kk in fill_methods] + ['Ours']
    methods_list = [f'{kk}+{tt}' for kk in fill_methods for tt in ['RF', 'SVM']]
    methods_list += [f'{kk}+GCN' for kk in fill_methods] + ['Ours']

    for lost in lost_nums[:1]:
        for rand in range(1, 3):
            res = Parallel(n_jobs=5)(delayed(contrastFind)(lost, mm, rand) for mm in methods_list)
            res = [kk for kk in res if kk is not None]
            if len(res) == 0:
                continue
            res = pd.concat(res, axis=0)
            res.insert(loc=0, column='lost_num', value=lost)
            os.makedirs(os.path.join(out_path, f'L{lost}', 'Contrast'), exist_ok=True)
            res.to_csv(os.path.join(out_path, f'L{lost}', 'Contrast', f'rand{rand}.csv'), index=False)


def contrastFind(lost_num, method, rand):
    root_dir = os.path.join(out_path, f'L{lost_num}')
    if re.search('[+].*', method):
        if re.search('[+].*', method).group() == '+GCN':
            file_name = f'CV5ave_accMax_{method}.csv'
        elif re.search('[+].*', method).group() in ['+RF', '+SVM']:
            file_name = f'CV5ave_one_{method}.csv'
        else:
            raise ValueError(method)
    else:
        assert method == 'Ours'
        file_name = 'CV5ave_accMax_Ours.csv'

    if os.path.isfile(os.path.join(root_dir, file_name)):
        df = pd.read_csv(os.path.join(root_dir, file_name))
        dfi = df[df['rand'] == rand]
        dfi.insert(loc=0, column='method', value=method)
        return dfi
    else:
        return None


def concatRandOurs(lost_num, MaxMet='acc'):
    root_dir = os.path.join(root_path, '20230628-ADratio')
    out_dir = os.path.join(out_path, f'L{lost_num}')
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isfile(os.path.join(out_dir, f'CV5ave_{MaxMet}Max_Ours.pkl')):
        with open(os.path.join(out_dir, f'CV5ave_{MaxMet}Max_Ours.pkl'), 'rb') as f:
            evals_dict = pickle.load(f)
    else:
        evals_dict = dict()

    rand_list = sorted([int(kk.split('rand')[1]) for kk in os.listdir(root_dir)])
    for rand in rand_list:
        for para_name in sorted(os.listdir(os.path.join(root_dir, f'rand{rand}', f'L{lost_num}', 'fold0'))):
            para_name0 = '__'.join(para_name.split('__')[1:])
            if f'{para_name0}_R{rand}' in evals_dict.keys():
                continue
            evals = Ave5foldMax(os.path.join(root_dir, f'rand{rand}', f'L{lost_num}'), para_name0, MaxMet)
            if evals is None:
                continue
            evals_dict.update({f'{para_name0}_R{rand}': evals[1]})

    with open(os.path.join(out_dir, f'CV5ave_{MaxMet}Max_Ours.pkl'), 'wb') as f:
        pickle.dump(evals_dict, f)
    evals_df = []
    for ii, vv in evals_dict.items():
        # print(vv)
        evals_df.append([int(ii.split('_R')[1]), ii.split('_R')[0]] + list(vv))
    evals_df = pd.DataFrame(evals_df, columns=['rand', 'para_name', 'epoch'] + metrics)
    evals_df.sort_values(['rand'], inplace=True)
    evals_df.to_csv(os.path.join(out_dir, f'CV5ave_{MaxMet}Max_Ours.csv'), index=False)


def concatRandGCN(lost_num, impute, MaxMet='acc'):
    method = f'{impute}+GCN'
    root_dir = os.path.join(root_path, '20230703-ImputeGCN', 'arti_rand', impute)
    out_dir = os.path.join(out_path, f'L{lost_num}')
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isfile(os.path.join(out_dir, f'CV5ave_{MaxMet}Max_{method}.pkl')):
        with open(os.path.join(out_dir, f'CV5ave_{MaxMet}Max_{method}.pkl'), 'rb') as f:
            evals_dict = pickle.load(f)
    else:
        evals_dict = dict()

    rand_list = sorted([int(kk.split('rand')[1]) for kk in os.listdir(root_dir)])
    for rand in rand_list:
        for para_name in sorted(os.listdir(os.path.join(root_dir, f'rand{rand}', f'L{lost_num}', 'fold0'))):
            para_name0 = '__'.join(para_name.split('__')[1:])
            if f'{para_name0}_R{rand}' in evals_dict.keys():
                continue
            evals = Ave5foldMax(os.path.join(root_dir, f'rand{rand}', f'L{lost_num}'), para_name0, MaxMet)
            if evals is None:
                continue
            evals_dict.update({f'{para_name0}_R{rand}': evals[1]})

    with open(os.path.join(out_dir, f'CV5ave_{MaxMet}Max_{method}.pkl'), 'wb') as f:
        pickle.dump(evals_dict, f)
    evals_df = []
    for ii, vv in evals_dict.items():
        print(vv)
        evals_df.append([int(ii.split('_R')[1]), ii.split('_R')[0]] + list(vv))
    evals_df = pd.DataFrame(evals_df, columns=['rand', 'para_name', 'epoch'] + metrics)
    evals_df.sort_values(['rand'], inplace=True)
    evals_df.to_csv(os.path.join(out_dir, f'CV5ave_{MaxMet}Max_{method}.csv'), index=False)


def Ave5foldMax(path5f, para_name0, met='acc'):
    metrics_5CV = []
    for fold in range(5):
        dir0 = os.path.join(path5f,  f'fold{fold}')
        if not os.path.exists(dir0):
            print(f'{dir0} not exist')
            return None
        ff = [kk for kk in os.listdir(dir0) if kk.endswith(para_name0)]
        if len(ff) != 1:
            print(f'{dir0}, {para_name0} not exist')
            return None
        if not os.path.isfile(os.path.join(dir0, ff[0], 'test_indicators.csv')):
            print(f'{dir0}, {para_name0} not complete')
            return None
        dfi = pd.read_csv(os.path.join(dir0, ff[0], 'test_indicators.csv'))
        metrics_5CV.append(dfi.values)  # stack后(5,250,5)
    ave_5CV = pd.DataFrame(np.mean(np.stack(metrics_5CV), axis=0), columns=metrics)  # (250,5)
    ave_5CV.reset_index(drop=False, inplace=True)
    acc_id, acc_max = FindMax(met='acc', ave=ave_5CV)
    print(f'Max {met}: {acc_max},\tepoch: {acc_id}')
    return ave_5CV, ave_5CV.iloc[acc_id, :]


def FindMax(met, ave):
    values = ave[met].values
    met_max = values.max()  # 5折平均后的最大值
    met_id = np.argmax(values, axis=0)
    return met_id, met_max


def concatRandRFSVM(lost_num, impute, clf='RF'):
    method = f'{impute}+{clf}'
    root_dir = os.path.join(root_path, f'20230629-Impute{clf}', 'arti_rand', impute)
    out_dir = os.path.join(out_path, f'L{lost_num}')
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isfile(os.path.join(out_dir, f'CV5ave_one_{method}.pkl')):
        with open(os.path.join(out_dir, f'CV5ave_one_{method}.pkl'), 'rb') as f:
            evals_dict = pickle.load(f)
    else:
        evals_dict = dict()

    rand_list = sorted([int(kk.split('rand')[1]) for kk in os.listdir(root_dir)])
    for rand in rand_list:
        if f'{method}_R{rand}' in evals_dict.keys():
            continue
        evals = Ave5foldCLF(os.path.join(root_dir, f'rand{rand}', f'L{lost_num}'))
        if evals is None:
            continue
        evals_dict.update({f'{method}_R{rand}': evals})

    with open(os.path.join(out_dir, f'CV5ave_one_{method}.pkl'), 'wb') as f:
        pickle.dump(evals_dict, f)
    evals_df = []
    for ii, vv in evals_dict.items():
        evals_df.append([int(ii.split('_R')[1]), ii.split('_R')[0]] + list(vv))
    evals_df = pd.DataFrame(evals_df, columns=['rand', 'para_name', 'epoch'] + metrics)
    evals_df.sort_values(['rand'], inplace=True)
    evals_df.to_csv(os.path.join(out_dir, f'CV5ave_one_{method}.csv'), index=False)


def Ave5foldCLF(path5f):
    metrics_5CV = []
    for fold in range(5):
        dir0 = os.path.join(path5f,  f'fold{fold}')
        if not os.path.exists(dir0):
            return None
        dfi = pd.read_csv(os.path.join(dir0, 'test_indicators.csv'), index_col=0)
        metrics_5CV.append(dfi.values)  # stack后(5,1,5)
    ave_5CV = pd.DataFrame(np.mean(np.stack(metrics_5CV), axis=0), columns=metrics)  # (1,5)
    ave_5CV.reset_index(drop=False, inplace=True)
    return ave_5CV.iloc[-1, :]


def CorrectMetrics(rand, lost, fold):
    dir0 = os.path.join(root_path, f'rand{rand}', f'L{lost}', f'fold{fold}')
    for para_name in sorted(os.listdir(dir0)):
        predictions = pd.read_csv(os.path.join(dir0, para_name, 'test_predictions.csv'))
        true = predictions.iloc[:, 0]
        indicators = []
        for i in range(250):
            prob = predictions.iloc[:, [i*3+1, i*3+2, i*3+3]]
            auc, _, _ = get_auc(true=true, prob=prob)
            evals = eval_metric(true=true, prob=prob)
            df = pd.DataFrame([[evals[ave][kk] for kk in metrics[:-1]] + [auc[ave]]],
                              columns=[kk if kk[:3] == 'acc' else f'{kk}_{ave}' for kk in metrics], index=[0])
            indicators.append(df)
        indicators = pd.concat(indicators)
        indicators.to_csv(os.path.join(dir0, para_name, 'test_indicators.csv'), index=False)
        if os.path.isfile(os.path.join(dir0, para_name, 'test_indicators1.csv')):
            os.system('rm ' + os.path.join(dir0, para_name, 'test_indicators1.csv'))


def Concat5fold(path5f, para_name0):
    scores_5CV = pd.DataFrame()
    flag = 0
    for fold in range(5):
        dir0 = os.path.join(path5f,  f'fold{fold}')
        if not os.path.exists(dir0):
            break
        ff = [kk for kk in os.listdir(dir0) if kk.endswith(para_name0)]
        if len(ff) != 1:
            print(f'{dir0}, {para_name0} not exist')
            break
        dfi = pd.read_csv(os.path.join(dir0, ff[0], 'test_predictions.csv'))
        score_i = pd.DataFrame({
            'true': dfi.iloc[:, 0],
            'prob1': dfi.iloc[:, -1],
        })
        scores_5CV = pd.concat([scores_5CV, score_i], axis=0)
        flag += 1
    if flag < 5:
        return None
    prob = np.hstack([1 - scores_5CV[['prob1']].values, scores_5CV[['prob1']].values])
    indicators = evaluate(prob, true=scores_5CV[['true']].values)
    return indicators


def Ave5fold(path5f, para_name0):
    metrics_5CV = []
    flag = 0
    for fold in range(5):
        dir0 = os.path.join(path5f,  f'fold{fold}')
        if not os.path.exists(dir0):
            break
        ff = [kk for kk in os.listdir(dir0) if kk.endswith(para_name0)]
        if len(ff) != 1:
            print(f'{dir0}, {para_name0} not exist')
            return None
        dfi = pd.read_csv(os.path.join(dir0, ff[0], 'test_indicators.csv'))
        metrics_5CV.append(dfi.iloc[-1, :].values)
        flag += 1
    if flag < 5:
        return None
    indicators = pd.DataFrame(np.mean(np.vstack(metrics_5CV), axis=0, keepdims=True), columns=metrics)
    return indicators


if __name__ == '__main__':
    main()
