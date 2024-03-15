from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
import random
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from utils import read_data, eval_metric, get_auc, eval_metric_cl2, get_auc_cl2


def main():
    # Training settings 训练参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./../../results/')
    parser.add_argument('--data_root', type=str, default='./../../data/')
    parser.add_argument('--data_name', type=str, default='nat1')
    parser.add_argument('--rand', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--fold', type=int, default=0, help='fold index, [0,1,2,3,4]')
    parser.add_argument('--fill', type=str, default='', help='method for fill NaN')
    parser.add_argument('--lost_name', type=str, default='L999', help='number of missing features')
    parser.add_argument('--classify', type=str, default='', choices=['RF', 'SVM'])
    parser.add_argument('--seed', type=int, default=2021, help='random seed (default: 2021)')

    args = parser.parse_args()

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)

    if (args.data_name.startswith('nat') and 'drop' in args.data_name) or args.data_name == 'full':
        assert args.fill == '' and args.lost_name == 'L999'
        out_dir = os.path.join(args.save_dir, f'Impute{args.classify}', args.data_name,
                               f'rand{args.rand}', args.lost_name, f's{args.seed}', f'fold{args.fold}')
    else:
        out_dir = os.path.join(args.save_dir, f'Impute{args.classify}', args.data_name, args.fill,
                               f'rand{args.rand}', args.lost_name, f's{args.seed}', f'fold{args.fold}')
    os.makedirs(out_dir, exist_ok=True)

    train_x, train_y, test_x, test_y = load_data(args)
    # print(train_x.shape, test_x.shape)

    if args.classify == 'RF':
        clf = RandomForestClassifier(random_state=args.seed, n_estimators=100)
        clf.fit(train_x, train_y)
    elif args.classify == 'SVM':
        clf = svm.SVC(kernel='rbf', probability=True, random_state=args.seed)
        clf.fit(train_x, train_y)
    else:
        raise ValueError(args.classify)

    # train
    prob_tr = clf.predict_proba(train_x)
    # pred_tr = clf.predict(train_x)
    probs_tr, df_tr = evaluate(prob_tr, train_y)

    prob_te = clf.predict_proba(test_x)
    # pred_te = clf.predict(test_x)
    probs_te, df_te = evaluate(prob_te, test_y)

    # save
    probs_tr.to_csv(os.path.join(out_dir, 'train_label.prob.pred.csv'))
    df_tr.to_csv(os.path.join(out_dir, 'train_indicators.csv'))
    probs_te.to_csv(os.path.join(out_dir, 'test_label.prob.pred.csv'))
    df_te.to_csv(os.path.join(out_dir, 'test_indicators.csv'))


def load_data(args):
    if args.data_name in ['BC', 'CC', 'CK', 'HC', 'HD', 'HP', 'HS', 'PI']:
        assert args.lost_name == 'L999'
        if args.fill == 'random':
            data_dir = os.path.join(args.data_root, 'uci', args.data_name, f'divide_{args.rand}')
            data_tr, data_te = read_data(data_dir, args.fold, rd=True)
        else:
            data_dir = os.path.join(args.data_root, 'imputed_data', args.data_name, args.fill, f'rand{args.rand}', args.lost_name)
            data_tr, data_te = read_data(data_dir, args.fold, rd=False)

    elif args.data_name in ['nat1', 'nat2']:
        assert args.lost_name == 'L999'
        if args.fill == 'random':
            data_dir = os.path.join(args.data_root, 'nat', f'{args.data_name}_0.8_', f'divide_{args.rand}')
            data_tr, data_te = read_data(data_dir, args.fold, rd=True)
        else:
            data_dir = os.path.join(args.data_root, 'imputed_data', args.data_name, args.fill, f'rand{args.rand}', args.lost_name)
            data_tr, data_te = read_data(data_dir, args.fold, rd=False)

    elif args.data_name in ['nat1_dropFeature', 'nat1_dropSample', 'nat2_dropFeature']:
        assert args.fill == '' and args.lost_name == 'L999'
        data_dir = os.path.join(args.data_root, 'nat', args.data_name + '_', f'divide_{args.rand}')
        data_tr = pd.read_csv(os.path.join(data_dir, f'index_data_label_{args.fold}_train.csv'), header=None)
        data_te = pd.read_csv(os.path.join(data_dir, f'index_data_label_{args.fold}_test.csv'), header=None)

    elif args.data_name in ['MCAR', 'MAR', 'MNAR']:
        assert args.lost_name != 'L999'
        if args.fill == 'random':
            data_dir = os.path.join(args.data_root, 'simu', f'simu_1100_{args.data_name}', f'divide_{args.rand}', args.lost_name)
            data_tr, data_te = read_data(data_dir, args.fold, rd=True)
        else:
            data_dir = os.path.join(args.data_root, 'imputed_data', args.data_name, args.fill, f'rand{args.rand}', args.lost_name)
            data_tr, data_te = read_data(data_dir, args.fold, rd=False)

    elif args.data_name == 'full':
        assert args.fill == '' and args.lost_name == 'L999'
        data_dir = os.path.join(args.data_root, 'simu', 'simu_1100_full', f'divide_{args.rand}')
        data_tr = pd.read_csv(os.path.join(data_dir, f'index_data_label_full_{args.fold}_train.csv'), header=None)
        data_te = pd.read_csv(os.path.join(data_dir, f'index_data_label_full_{args.fold}_test.csv'), header=None)

    else:
        print('!!!wrong input: data_name')
        raise ValueError
    # ------------------------------------------

    if args.data_name in ['nat1_dropFeature', 'nat1_dropSample', 'nat2_dropFeature']:
        labels_tr = data_tr.iloc[:, -1].values
        labels_te = data_te.iloc[:, -1].values
        data_tr = data_tr.iloc[:, 1:-1]
        data_te = data_te.iloc[:, 1:-1]
    else:
        labels_tr = data_tr.iloc[:, -2].values
        labels_te = data_te.iloc[:, -2].values
        data_tr = data_tr.iloc[:, 1:-2]
        data_te = data_te.iloc[:, 1:-2]
    # print(np.where(np.isnan(data_te)))

    return data_tr, labels_tr, data_te, labels_te


def evaluate(prob, true):
    pred = np.argmax(prob, axis=1)
    num_class = int(max(true) + 1)
    probs = np.hstack((np.expand_dims(true, axis=1), prob, np.expand_dims(pred, axis=1)))
    probs = pd.DataFrame(probs, columns=['true'] + [f'pr{kk}' for kk in range(num_class)] + ['pred'])
    metrics = ['acc', 'pre', 'sen', 'spe', 'f1', 'auc']
    if num_class == 2:
        auc, _, _ = get_auc_cl2(true=true, prob=prob)
        evals = eval_metric_cl2(true=true, prob=prob)
        df = pd.DataFrame([[evals[kk] for kk in metrics[:-1]] + [auc]], columns=metrics, index=[0])
    elif num_class > 2:
        ave = 'macro'
        auc, _, _ = get_auc(true=true, prob=prob)
        evals = eval_metric(true=true, prob=prob)
        df = pd.DataFrame([[evals[ave][kk] for kk in metrics[:-1]] + [auc[ave]]],
            columns=[kk if kk[:3] == 'acc' else f'{kk}_{ave}' for kk in metrics], index=[0])
    else:
        raise ValueError
    return probs, df


if __name__ == '__main__':
    main()


