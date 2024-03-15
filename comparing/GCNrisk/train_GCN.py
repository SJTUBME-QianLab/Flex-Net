# Copyright (c) 2016 Thomas Kipf
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import division
from __future__ import print_function

import time
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
import os
import random
from gcn.utils import *
from gcn.models import MLP, GCN
from utils import eval_metric, get_auc, eval_metric_cl2, get_auc_cl2

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def get_train_test_masks(train_ind, test_ind, labels_onehot):
    mask_train = sample_mask(train_ind, labels_onehot.shape[0])
    mask_test = sample_mask(test_ind, labels_onehot.shape[0])
    y_train = np.zeros(labels_onehot.shape)
    y_test = np.zeros(labels_onehot.shape)
    y_train[mask_train, :] = labels_onehot[mask_train, :]  # true label for training set; [0,0,0] for test
    y_test[mask_test, :] = labels_onehot[mask_test, :]

    return y_train, y_test, mask_train, mask_test


def evaluate(prob, true_onehot):
    true = np.argmax(true_onehot, axis=1)
    num_class = true_onehot.shape[1]
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
    return df


def run_training(adj, features, labels_onehot, train_ind, test_ind, args):
    # Create test and train masked variables
    y_train, y_test, mask_train, mask_test = \
        get_train_test_masks(train_ind, test_ind, labels_onehot)

    # Some preprocessing
    features = preprocess_features(features)
    if args.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif args.model == 'cheby':  # ##用这个
        support = chebyshev_polynomials(adj, args.max_degree)
        num_supports = 1 + args.max_degree
        model_func = GCN
    elif args.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for GCN model ')
    
    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'phase_train': tf.placeholder_with_default(False, shape=()),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }
    
    # Create model
    args.learning_rate = args.lr
    model = model_func(placeholders, input_dim=features[2][1], logging=True, arg=args)
    
    # Initialize session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True  # allocate dynamically
    sess = tf.Session(config=config)
    # sess = tf.Session()

    # Define model evaluation function *************************
    def test(feats, graph, label, mask, placeholder):
        feed_dict_val = construct_feed_dict(feats, graph, label, mask, placeholder)
        feed_dict_val.update({placeholder['phase_train'].name: False})
        _, acci, predi = sess.run([model.loss, model.accuracy, model.predict()], feed_dict=feed_dict_val)

        # pred: (1446,3)
        predi = predi[np.squeeze(np.argwhere(mask == 1)), :]
        # label: (1446,3)
        lab_o = label[np.squeeze(np.argwhere(mask == 1)), :]  # true label onehot
        dfi = evaluate(predi, lab_o)
        lab = np.argmax(lab_o, axis=1)  # true label 012
        return dfi, lab, predi, acci
    
    # Init variables
    sess.run(tf.global_variables_initializer())

    df_test, df_train = [], []
    pred_test, pred_train = [], []
    train_loss = []

    # Train model
    for epoch in range(args.epochs):
    
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, mask_train, placeholders)
        feed_dict.update({placeholders['dropout']: args.dropout, placeholders['phase_train']: True})

        # Training step
        _, loss, acc_tr, pred_tr = sess.run(
            [model.opt_op, model.loss, model.accuracy, model.predict()],
            feed_dict=feed_dict
        )
        if epoch % args.log_interval == 0:
            print('Training epoch: {}'.format(epoch))
            print(f'\tTraining loss: {loss}.')
            print(f'\tTraining acc: {acc_tr}.')

        # pred: (1446,3)
        pred_tr = pred_tr[np.squeeze(np.argwhere(mask_train == 1)), :]
        # label: (1446,3)
        labs = y_train[np.squeeze(np.argwhere(mask_train == 1)), :]
        dfi_train = evaluate(pred_tr, labs)
        true_tr = np.argmax(labs, axis=1)

        # Validation
        dfi_test, true_te, pred_te, acc_test = test(
            features, support, y_test, mask_test, placeholders)

        if epoch % args.log_interval == 0:
            print('Eval epoch: {}'.format(epoch))
            print(f'\tTesting acc: {acc_test}.')

        train_loss.append(loss)
        df_train.append(dfi_train)
        pred_train.extend(pred_tr.T)
        df_test.append(dfi_test)
        pred_test.extend(pred_te.T)

    print("Optimization Finished!")

    return true_tr, pred_train, df_train, train_loss, true_te, pred_test, df_test
