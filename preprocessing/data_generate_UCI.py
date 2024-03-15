import pandas as pd
import numpy as np
import os
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./../data/')
    parser.add_argument('--data_name', type=str, default='BC')
    args = parser.parse_args()

    data1 = preprocess(args.data_name)

    out_dir1 = os.path.join(args.data_root, 'uci', args.data_name)
    os.makedirs(out_dir1, exist_ok=True)

    # statistic for missing patterns
    info, data_7 = define_loss(data1)
    nan = np.isnan(data_7.values[:, 1:-2])
    missing_f = np.sum(nan.sum(axis=0) > 0)
    missing_s = np.sum(nan.sum(axis=1) > 0)

    data_7.to_csv(os.path.join(out_dir1, 'index_data_label_lost.csv'), index=False, header=False)

    info_count = stat(info, missing_f, missing_s)
    with pd.ExcelWriter(os.path.join(out_dir1, 'info_count.xlsx')) as writer:
        info.to_excel(writer, sheet_name='all_info', index=False)
        info_count.to_excel(writer, sheet_name='all_count', index=False)

    with open(os.path.join(args.data_root, 'uci', 'log.txt'), 'a') as f:
        f.write("-------------------------------------\n")
        f.write(time.strftime("%Y-%m-%d %X", time.localtime()) + '\n')
        f.write('out dir:\t' + out_dir1 + '\n')
        f.write('loss pattern num:\t' + str(len(info_count) - 1) + '\n')
        f.write('sample num:\t' + str(data_7.shape[0]) + '\n')
        f.write('feature num:\t' + str(data_7.shape[1] - 3) + '\n')
        f.write('incomplete sample num:\t' + str(missing_s) + '\n')
        f.write('incomplete feature num:\t' + str(missing_f) + '\n')


def preprocess(data_name):
    raw_dir = './../raw_data/UCI'

    if data_name == 'BC':
        rawdata = pd.read_csv(os.path.join(raw_dir, 'Breast Cancer', 'breast.csv'), header=None)
        data1 = rawdata.replace('?', np.nan, inplace=False)
        data1.columns = [str(x + 1) for x in range(11)]
        # label
        label_col = '11'
        data1['label'] = data1[label_col].apply(lambda n: int(bool(n / 2 - 1)))
        # 2 (0) for benign, 4 (1) for malignant
        data1.drop([label_col], axis=1, inplace=True)

    elif data_name == 'CC':
        rawdata = pd.read_csv(os.path.join(raw_dir, 'Cervical Cancer', 'risk_factors_cervical_cancer.csv'))
        data1 = rawdata.replace('?', np.nan, inplace=False)
        # label
        label_col = 'Dx'
        data1['label'] = data1[label_col].values.astype(int)
        data1.drop(label_col, axis=1, inplace=True)

    elif data_name == 'CK':
        rawdata = pd.read_csv(os.path.join(raw_dir, 'Chronic Kidney Disease', 'kidney.csv'), header=0)
        data1 = rawdata.replace('?', np.nan, inplace=False)
        # nominal convert to int
        nominal = list(range(3, 10)) + list(range(19, 25))
        for kk in nominal:  # char convert to number
            data1.iloc[:, kk - 1] = pd.factorize(data1.iloc[:, kk - 1])[0].astype(np.uint16)
        data1.replace(np.uint16(-1), np.nan, inplace=True)
        # label
        label_dict = {'ckd': 0, 'notckd': 1}  # 0有病，1没病
        label_col = 'class'
        data1['label'] = data1[label_col].apply(lambda n: label_dict[n])
        data1.drop([label_col], axis=1, inplace=True)

    elif data_name == 'HC':
        rawdata = pd.read_csv(os.path.join(raw_dir, 'Horse Colic', 'horse.csv'), header=None)
        data1 = rawdata.replace('?', np.nan, inplace=False)
        data1.drop([2], axis=1, inplace=True)
        data1.columns = [str(x + 1) for x in range(23)]
        # label
        label_col = '23'
        data1['label'] = data1[label_col].apply(lambda n: int(bool(n - 1)))
        data1.drop([label_col], axis=1, inplace=True)

    elif data_name == 'HD':
        rawdata = pd.read_csv(os.path.join(raw_dir, 'heart-disease', 'reprocessed.hungarian.csv'), header=None)
        data1 = rawdata.replace(-9, np.nan, inplace=False)
        data1.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                         'slope', 'ca', 'thal', 'num']
        # label
        label_col = 'num'
        data1['label'] = data1[label_col].apply(lambda n: int(bool(n)))
        # -- Value 0: < 50% diameter narrowing
        # -- Value 1: > 50% diameter narrowing
        data1.drop([label_col], axis=1, inplace=True)

    elif data_name == 'HP':
        rawdata = pd.read_csv(os.path.join(raw_dir, 'Hepatitis', 'hepa.csv'), header=None)
        data1 = rawdata.replace('?', np.nan, inplace=False)
        data1.columns = ['Class', 'AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA', 'LIVER BIG',
                         'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'BILIRUBIN',
                         'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']
        # label
        label_col = 'Class'
        data1['label'] = data1[label_col].apply(lambda n: int(bool(n - 1)))
        # Class: DIE, LIVE; 1, 2 不知道对应
        data1.drop([label_col], axis=1, inplace=True)

    elif data_name == 'HS':
        rawdata = pd.read_csv(os.path.join(raw_dir, 'hcc-survival', 'hcc.csv'), header=None)
        data1 = rawdata.replace('?', np.nan, inplace=False)
        # label
        label_col = data1.shape[1] - 1
        data1['label'] = data1[label_col].values.astype(int)
        data1.drop([label_col], axis=1, inplace=True)

    elif data_name == 'PI':
        rawdata = pd.read_csv(os.path.join(raw_dir, 'pima-indians-diabetes', 'pima.csv'), header=None)
        data1 = rawdata.replace('nan', np.nan, inplace=False)
        # label
        label_col = data1.shape[1] - 1
        label_dict = {-1: 0, 1: 1}
        data1['label'] = data1[label_col].apply(lambda n: label_dict[n])
        data1.drop([label_col], axis=1, inplace=True)

    else:
        print('wrong data_name!')
        raise ValueError

    data1.iloc[:, :-1] = data1.iloc[:, :-1].values.astype(float)
    data1.reset_index(drop=False, inplace=True)

    return data1


def define_loss(data_5):
    data_6 = data_5.copy()
    loss = []

    for i in range(data_6.shape[0]):
        flag = ''
        for j in range(data_6.shape[1]):
            if np.isnan(data_6.iloc[i, j]):  # Y=Yes, missing
                flag = flag + 'Y'
            else:  # N=No, not missing, observed
                flag = flag + 'N'
        loss.append(flag)  # denote sample missing pattern using a string

    data_6['loss'] = loss
    data_6.sort_values(['loss'], inplace=True, kind='mergesort')
    data_6.reset_index(drop=True, inplace=True)
    # loss_flag: the index of different missing patterns
    data_6['loss_flag'] = np.nan
    loss_flag = 0
    target = data_6.loc[0, 'loss']
    for i in range(data_6.shape[0]):
        if data_6.loc[i, 'loss'] != target:
            loss_flag = loss_flag + 1
            target = data_6.loc[i, 'loss']
        data_6.loc[i, 'loss_flag'] = loss_flag

    info = data_6.iloc[:, [0] + list(range(data_6.shape[1] - 3, data_6.shape[1]))]
    info.columns = ['index', 'label', 'loss', 'loss_flag']
    data_7 = data_6.drop(['loss'], axis=1, inplace=False)

    return info, data_7


def stat(info, missing_f, missing_s):
    # count: the number of samples with the same missing pattern
    listname = ['count', 'count0', 'count1']
    info_count = info[['loss', 'loss_flag']].drop_duplicates().reset_index(drop=True)
    info_count['loss_num'] = info_count['loss'].apply(lambda x: x.count('Y'))
    tmp = []
    for i in range(len(info_count)):
        dfi = info[info['loss_flag'] == i]
        l0 = len(dfi[dfi['label'] == 0])
        l1 = len(dfi[dfi['label'] == 1])
        tmp.append([len(dfi), l0, l1])
    info_count = pd.concat([info_count, pd.DataFrame(tmp, columns=listname)], axis=1)

    info_count['sum'] = info_count['loss_num'] * info_count['count']
    loc = info_count.shape[0]
    info_count.loc[loc, listname] = np.sum(info_count.loc[:, listname], axis=0)
    feature_num = len(info_count.loc[0, 'loss']) - 2
    info_count.loc[loc, 'loss'] = missing_s
    info_count.loc[loc, 'loss_flag'] = missing_f
    info_count.loc[loc, 'loss_num'] = feature_num
    info_count.loc[loc, 'sum'] = info_count['sum'].sum() / (feature_num * info_count.loc[loc, 'count'])
    assert missing_s == info_count[info_count['loss_num'] > 0]['count'].values[:-1].sum()

    return info_count


if __name__ == '__main__':
    main()
