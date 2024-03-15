import pandas as pd
import numpy as np
import os
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./../data/nat/')
    parser.add_argument('--data_name', type=str, default='nat1')
    args = parser.parse_args()
    
    data_dict = {
        'nat1': 'nat1_0.8_',
        'nat2': 'nat2_0.8_',
    }
    data_name = data_dict[args.data_name]
    out_dir = os.path.join(args.data_root, 'nat', data_name).rstrip('/')
    refine(out_dir)


def refine(out_dir):
    fff = 0

    if os.path.isfile(out_dir + 'value_lost.csv'):
        data_7 = pd.read_csv(out_dir + 'value_lost.csv', header=None)
        data_7.rename(columns={data_7.shape[1]-1: 'loss_flag'}, inplace=True)
        info = pd.read_excel(out_dir + 'info_count.xlsx', sheet_name='all_info')
        info_count = pd.read_excel(out_dir + 'info_count.xlsx', sheet_name='all_count')
        info_count.drop(info_count.shape[0]-1, axis=0, inplace=True)
    else:
        fff = 1
        data_5 = pd.read_csv(out_dir + 'value.csv', header=None)
        # statistics for missing patterns
        info, data_7 = define_loss(data_5)
        data_7.to_csv(out_dir + 'value_lost.csv', index=False, header=None)
        # info_count = stat(info)
        info_count = stat_cl3(info)
        with pd.ExcelWriter(out_dir + 'info_count.xlsx') as writer:
            info.to_excel(writer, sheet_name='all_info', index=False)
            info_count.to_excel(writer, sheet_name='all_count', index=False)

    info_in, info_count_in, data_8 = filter_data(info, info_count, data_7)

    os.makedirs(out_dir, exist_ok=True)
    with pd.ExcelWriter(os.path.join(out_dir, 'info_count.xlsx')) as writer:
        info_in.to_excel(writer, sheet_name='part_info', index=False)
        info_count_in.to_excel(writer, sheet_name='part_count', index=False)
    writer.save()
    writer.close()
    data_8.to_csv(os.path.join(out_dir, 'index_data_label_lost.csv'), index=False, header=False)

    with open(os.path.join(os.path.split(out_dir)[0], 'log.txt'), 'a') as f:
        f.write("-------------------------------------\n")
        f.write(time.strftime("%Y-%m-%d %X", time.localtime()) + '\n')
        f.write('out dir:\t' + out_dir + '\n')
        if fff == 1:
            f.write('raw sample num:\t' + str(data_5.shape[0]) + '\n')
            f.write('raw loss num:\t' + str(len(info_count) - 1) + '\n')
        f.write('tag:\t' + '\n')
        f.write('sample num:\t' + str(data_8.shape[0]) + '\n')
        f.write('loss num:\t' + str(len(info_count_in) - 1) + '\n')
        f.write('feature num:\t' + str(data_8.shape[1] - 3) + '\n')


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


def stat_cl3(info):
    # count: the number of samples with the same missing pattern
    listname = ['count', 'count0', 'count1', 'count2']
    info_count = info[['loss', 'loss_flag']].drop_duplicates().reset_index(drop=True)
    info_count['loss_num'] = info_count['loss'].apply(lambda x: x.count('Y'))
    tmp = []
    for i in range(len(info_count)):
        dfi = info[info['loss_flag'] == i]
        l0 = len(dfi[dfi['label'] == 0])
        l1 = len(dfi[dfi['label'] == 1])
        l2 = len(dfi[dfi['label'] == 2])
        tmp.append([len(dfi), l0, l1, l2])
    info_count = pd.concat([info_count, pd.DataFrame(tmp, columns=listname)], axis=1)

    info_count.loc[info_count.shape[0], listname] = np.sum(info_count.loc[:, listname], axis=0)

    return info_count


def filter_data(info, info_count, data_7):
    info_count_in = []
    data_new = pd.DataFrame()
    for i in range(len(info_count)):
        dfi = data_7[data_7['loss_flag'] == i]
        dfi0 = dfi[dfi.iloc[:, -2] == 0]
        dfi1 = dfi[dfi.iloc[:, -2] == 1]
        dfi2 = dfi[dfi.iloc[:, -2] == 2]
        if len(dfi0) < 2 or len(dfi1) < 2 or len(dfi2) < 2:
            continue
        info_count_in.append(info_count.iloc[i, :].tolist())
        data_new = pd.concat([data_new, dfi], axis=0)

    info_count_in = pd.DataFrame(info_count_in, columns=info_count.columns)
    info_count_in['new_loss_flag'] = list(range(len(info_count_in)))
    data_new = pd.merge(data_new, info_count_in[['loss_flag', 'new_loss_flag']]).drop(['loss_flag'], axis=1)
    info_in = pd.merge(info, data_new[[0, 'new_loss_flag']], left_on='index', right_on=0).drop(['loss_flag', 0], axis=1)

    nan = np.isnan(data_new.values[:, 1:-2])
    missing_f = np.sum(nan.sum(axis=0) > 0)
    missing_s = np.sum(nan.sum(axis=1) > 0)
    info_count_in['sum'] = info_count_in['loss_num'] * info_count_in['count']
    loc = info_count_in.shape[0]
    listname = ['count', 'count0', 'count1', 'count2']
    info_count_in.loc[loc, listname] = np.sum(info_count_in.loc[:, listname], axis=0)
    feature_num = len(info_count_in.loc[0, 'loss']) - 2
    info_count_in.loc[loc, 'loss'] = missing_s
    info_count_in.loc[loc, 'loss_flag'] = missing_f
    info_count_in.loc[loc, 'loss_num'] = feature_num
    info_count_in.loc[loc, 'sum'] = info_count_in['sum'].sum() / (feature_num * info_count_in.loc[loc, 'count'])
    assert missing_s == info_count_in[info_count_in['loss_num'] > 0]['count'].values[:-1].sum()

    return info_in, info_count_in, data_new


if __name__ == '__main__':
    main()
