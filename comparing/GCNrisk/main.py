from __future__ import division
from __future__ import print_function
import os
import time
import yaml
import pickle
from shutil import copytree, ignore_patterns
import argparse
import numpy as np
import pandas as pd
import random
from scipy import sparse
from scipy.spatial import distance
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import train_GCN as Train
from utils import read_data


def save_arg(args):
    losses = '{}.{}.{:d}.dr{:g}dec{:g}h{:d}lr{:g}e{:d}s{:d}'.format(
        args.model, args.sim, args.max_degree,
        args.dropout, args.decay, args.hidden1, args.lr, args.epochs, args.seed)
    args.exp_name = '__'.join([losses, args.exp_name])
    model_name = 't{}__{}'.format(time.strftime('%Y%m%d%H%M%S'), args.exp_name)

    if (args.data_name.startswith('nat') and 'drop' in args.data_name) or args.data_name == 'full':
        assert args.fill == '' and args.lost_name == 'L999'
        work_dir = os.path.join(args.save_dir, 'GCNrisk', args.data_name,
                                f'rand{args.rand}', 'L999', f'fold{args.fold}', model_name)
    else:
        work_dir = os.path.join(args.save_dir, 'GCNrisk', args.data_name, args.fill,
                                f'rand{args.rand}', args.lost_name, f'fold{args.fold}', model_name)
    os.makedirs(work_dir, exist_ok=True)
    print(args.exp_name)

    # copy all files
    pwd = os.path.dirname(os.path.realpath(__file__))
    copytree(pwd, os.path.join(work_dir, 'code'), symlinks=False, ignore=ignore_patterns('__pycache__'))
    arg_dict = vars(args)
    with open(os.path.join(work_dir, 'config.yaml'), 'w') as f:
        yaml.dump(arg_dict, f)
    return work_dir


def main():
    # Training settings 训练参数设置
    import warnings
    warnings.filterwarnings("ignore")
    # Training settings 训练参数设置
    parser = argparse.ArgumentParser(description='Graph CNNs for population graphs: TADPOLE dataset classification')
    parser.add_argument('--save_dir', type=str, default='./../../results/')
    parser.add_argument('--data_root', type=str, default='./../../data/')
    parser.add_argument('--data_name', type=str, default='nat1_dropFeature')
    parser.add_argument('--rand', type=int, default=1, help='use which data group')
    parser.add_argument('--seed', default=231, type=int, help='Seed for random initialisation (default: 231)')
    parser.add_argument('--fold', type=int, default=0, help='fold index, [0,1,2,3,4]')
    parser.add_argument('--fill', type=str, default='', help='method for fill NaN')
    parser.add_argument('--lost_name', type=str, default='L999', help='number of missing features')

    parser.add_argument('--exp_name', type=str, default='', help='Name of the experiment')
    parser.add_argument('--model', default='cheby', help='gcn model used (default: cheby, uses chebyshev polynomials, '
                                                         'options: gcn, cheby, dense )')
    parser.add_argument('--sim', default='ori', help='Type of similarity used for network.')
    parser.add_argument('--max_degree', default=3, type=int, help='Maximum Chebyshev polynomial degree.')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate (1 - keep probability)')  # 0.02
    parser.add_argument('--decay', default=1e-5, type=float, help='Weight for L2 loss on embedding matrix')
    parser.add_argument('--hidden1', default=64, type=int, help='Number of filters in hidden layers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate (default: 0.005,0.01)')
    parser.add_argument('--epochs', default=400, type=int, help='Number of epochs to train')
    parser.add_argument('--log_interval', type=int, default=5, help='wait epochs before logging training status')

    args = parser.parse_args()

    import fwr13y.d9m.tensorflow as tf_determinism

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    tf_determinism.enable_determinism()

    work_dir = save_arg(args)

    # ------------------------------------------
    data_tr, data_te = load_data(args)
    data_tr.columns = data_te.columns
    data_sort = pd.concat([data_tr, data_te], axis=0).reset_index(drop=True)
    data = data_sort.sample(frac=1, random_state=args.seed+args.fold)
    train_ind = [data.index.get_loc(x) for x in range(len(data_tr))]
    test_ind = [data.index.get_loc(x) for x in range(len(data_tr), len(data_sort))]
    # ------------------------------------------

    labels = data.iloc[:, -1].values
    features = data.iloc[:, 5:-1].values
    
    labels_onehot = pd.get_dummies(labels).values  # one-hot
    print(labels[:20])

    graph = constructGraph(features, args.sim)
    graph_use = [1, 2, 3, 4]  # age, sex, edu, apoe4
    phenotype = PhenotypeGraph(graph_use, data[graph_use], len(labels))
    graph = graph * phenotype
    
    # Classification with GCNs
    true_tr, pred_train, df_train, train_loss, true_te, pred_test, df_test = Train.run_training(
        graph, sparse.coo_matrix(features).tolil(), labels_onehot, train_ind, test_ind, args)  # train_GCN.py

    # save
    all_pred_tr = pd.DataFrame(np.hstack((np.array([true_tr]).T, np.array(pred_train).T)))
    all_pred_tr.to_csv(os.path.join(work_dir, 'train_predictions.csv'), index=False)
    df_train = pd.concat(df_train, axis=0)
    df_train['loss'] = train_loss
    df_train.to_csv(os.path.join(work_dir, 'train_indicators.csv'), index=False)

    all_pred_te = pd.DataFrame(np.hstack((np.array([true_te]).T, np.array(pred_test).T)))
    all_pred_te.to_csv(os.path.join(work_dir, 'test_predictions.csv'), index=False)
    df_test = pd.concat(df_test, axis=0)
    df_test.to_csv(os.path.join(work_dir, 'test_indicators.csv'), index=False)


def load_data(args):
    if args.data_name in ['nat1_dropFeature', 'nat1_dropSample', 'nat2_dropFeature']:
        assert args.fill == '' and args.lost_name == 'L999'
        data_dir = os.path.join(args.data_root, 'nat', args.data_name + '_with4risk', f'divide_{args.rand}')
        data_tr = pd.read_csv(os.path.join(data_dir, f'index_data_label_{args.fold}_train.csv'), header=None)
        data_te = pd.read_csv(os.path.join(data_dir, f'index_data_label_{args.fold}_test.csv'), header=None)

    else:
        print('!!!wrong input: data_name')
        raise ValueError

    return data_tr, data_te


def constructGraph(features, sim):
    if sim == 'ori':
        # Calculate all pairwise distances
        distv = distance.pdist(features, metric='correlation')  #
        # Convert to a square symmetric distance matrix
        dist = distance.squareform(distv)
        sigma = np.mean(dist)
        # Get affinity from similarity matrix
        graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    elif sim == 'gauStd':
        # Calculate all pairwise distances
        distv = distance.pdist(features, metric='correlation')  #
        # Convert to a square symmetric distance matrix
        dist = distance.squareform(distv)
        sigma = np.std(dist)
        # Get affinity from similarity matrix
        graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    else:
        raise ValueError(args.sim)
    return graph


def PhenotypeGraph(scores, feature_pheno, num_nodes):
    # [1, 2, 3, 4]  # age, sex, edu, apoe4
    graph = np.zeros((num_nodes, num_nodes))
    for l in scores:
        label_dict = feature_pheno[l]
        # print(label_dict[:5])
        if l in [1, 3]:  # age, edu
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict.values[k]) - float(label_dict.values[j]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass
        else:  # sex, apoe4
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict.values[k] == label_dict.values[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1
    return graph


if __name__ == "__main__":
    main()

