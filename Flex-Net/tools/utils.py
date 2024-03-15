import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc
from scipy import interp


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def onehot_code(Y, num_class):
    Y = np.array(Y)
    Yc_onehot = np.zeros((len(Y), num_class))
    for i in range(num_class):
        Yc_onehot[np.where(Y == i)[0], i] = 1.0
    return Yc_onehot


def eval_metric_cl2(true, prob):
    num_class = int(max(true) + 1)
    if isinstance(prob, pd.DataFrame):
        prob = prob.values
    if num_class == prob.shape[0]:  # num_class*num_sample -> num_sample*num_class
        prob = prob.T
    assert len(true) == prob.shape[0]
    pred = list(prob.argmax(axis=0 if num_class == prob.shape[0] else 1))
    assert len(true) == len(pred)

    # acc
    acc = sum([pred[i] == true[i] for i in range(len(true))]) / len(true)

    # confusion matrix
    con_matrix = np.array(
        [[sum([pred[i] == k1 and true[i] == k2 for i in range(len(true))]) for k1 in range(num_class)] for k2 in range(num_class)])
    tn, fp, fn, tp = con_matrix.ravel()
    con_arr = con_matrix.ravel()

    SEN_cal = lambda tn, fp, fn, tp: tp / (tp + fn) if (tp + fn) != 0 else 0
    PRE_cal = lambda tn, fp, fn, tp: tp / (tp + fp) if (tp + fp) != 0 else 0
    SPE_cal = lambda tn, fp, fn, tp: tn / (tn + fp) if (tn + fp) != 0 else 0
    NPV_cal = lambda tn, fp, fn, tp: tn / (tn + fn) if (tn + fn) != 0 else 0

    sen = SEN_cal(*con_arr)
    pre = PRE_cal(*con_arr)
    spe = SPE_cal(*con_arr)
    npv = NPV_cal(*con_arr)
    f1 = 2 * pre * sen / (pre + sen)

    evals = {
        'confusion_matrix': con_matrix,
        'acc': acc, 'accuracy': acc,
        'pre': pre, 'precision': pre, 'ppv': pre,
        'npv': npv,
        'sen': sen, 'sensitivity': sen, 'recall': sen, 'tpr': sen,
        'spe': spe, 'specificity': spe, 'tnr': spe,
        'fpr': 1-spe,
        'f1': f1, 'f1_score': f1, 'f1score': f1,
    }

    return evals


def get_auc_cl2(true, prob):
    num_class = int(max(true) + 1)
    assert num_class == 2
    if isinstance(prob, pd.DataFrame):
        prob = prob.values
    if num_class == prob.shape[0]:  # num_class*num_sample -> num_sample*num_class
        prob = prob.T
    assert len(true) == prob.shape[0]
    # y_label = onehot_code(true, num_class)
    prob = prob / np.sum(prob, axis=1, keepdims=True).repeat(prob.shape[-1], axis=1)

    fpr, tpr, _ = roc_curve(true, prob[:, 1])
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr


def eval_metric(true, prob):
    num_class = int(max(true) + 1)
    if isinstance(prob, pd.DataFrame):
        prob = prob.values
    if num_class == prob.shape[0]:  # num_class*num_sample -> num_sample*num_class
        prob = prob.T
    assert len(true) == prob.shape[0]
    pred = list(prob.argmax(axis=0 if num_class == prob.shape[0] else 1))
    assert len(true) == len(pred)

    # acc
    acc = sum([pred[i] == true[i] for i in range(len(true))]) / len(true)

    # confusion matrix
    con_matrix = np.array(
        [[sum([pred[i] == k1 and true[i] == k2 for i in range(len(true))]) for k1 in range(num_class)] for k2 in range(num_class)])

    con_arr = np.zeros((num_class, 4))
    for k in range(num_class):
        tp = sum([pred[i] == k and true[i] == k for i in range(len(true))])
        fp = sum([pred[i] == k and true[i] != k for i in range(len(true))])
        tn = sum([pred[i] != k and true[i] != k for i in range(len(true))])
        fn = sum([pred[i] != k and true[i] == k for i in range(len(true))])
        # print(tn, fp, fn, tp)
        con_arr[k, :] = [tn, fp, fn, tp]

    SEN_cal = lambda tn, fp, fn, tp: tp / (tp + fn) if (tp + fn) != 0 else 0
    PRE_cal = lambda tn, fp, fn, tp: tp / (tp + fp) if (tp + fp) != 0 else 0
    SPE_cal = lambda tn, fp, fn, tp: tn / (tn + fp) if (tn + fp) != 0 else 0
    NPV_cal = lambda tn, fp, fn, tp: tn / (tn + fn) if (tn + fn) != 0 else 0

    # macro
    sen = np.nansum(np.array([SEN_cal(*cc) for cc in con_arr])) / num_class
    pre = np.nansum(np.array([PRE_cal(*cc) for cc in con_arr])) / num_class
    spe = np.nansum(np.array([SPE_cal(*cc) for cc in con_arr])) / num_class
    npv = np.nansum(np.array([NPV_cal(*cc) for cc in con_arr])) / num_class
    f1 = 2 * pre * sen / (pre + sen)

    # micro
    sen_mi = SEN_cal(*list(np.sum(con_arr, axis=0)))
    pre_mi = PRE_cal(*list(np.sum(con_arr, axis=0)))
    spe_mi = SPE_cal(*list(np.sum(con_arr, axis=0)))
    npv_mi = NPV_cal(*list(np.sum(con_arr, axis=0)))
    f1_mi = 2 * pre_mi * sen_mi / (pre_mi + sen_mi)

    evals = {
        'macro': {
            'confusion_matrix': con_matrix,
            'acc': acc, 'accuracy': acc,
            'pre': pre, 'precision': pre, 'ppv': pre,
            'npv': npv,
            'sen': sen, 'sensitivity': sen, 'recall': sen, 'tpr': sen,
            'spe': spe, 'specificity': spe, 'tnr': spe,
            'fpr': 1-spe,
            'f1': f1, 'f1_score': f1, 'f1score': f1,
        },
        'micro': {
            'confusion_matrix': con_matrix,
            'acc': acc, 'accuracy': acc,
            'pre': pre_mi, 'precision': pre_mi, 'ppv': pre_mi,
            'npv': npv_mi,
            'sen': sen_mi, 'sensitivity': sen_mi, 'recall': sen_mi, 'tpr': sen_mi,
            'spe': spe_mi, 'specificity': spe_mi, 'tnr': spe_mi,
            'fpr': 1-spe_mi,
            'f1': f1_mi, 'f1_score': f1_mi, 'f1score': f1_mi,
        }
    }

    return evals


def get_auc(true, prob):
    num_class = int(max(true) + 1)
    if isinstance(prob, pd.DataFrame):
        prob = prob.values
    if num_class == prob.shape[0]:  # num_class*num_sample -> num_sample*num_class
        prob = prob.T
    assert len(true) == prob.shape[0]
    y_label = onehot_code(true, num_class)
    prob = prob / np.sum(prob, axis=1, keepdims=True).repeat(prob.shape[-1], axis=1)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # roc_auc_sk = roc_auc_score(true, prob, average='macro', multi_class='ovo')
    # print(roc_auc['macro'], roc_auc_sk)
    return roc_auc, fpr, tpr

