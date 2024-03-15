import pandas as pd
import numpy as np
import os
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./../data/')
    parser.add_argument('--data_name', type=str, default='nat1')
    args = parser.parse_args()

    data_dict = {
        'nat1': ['nat1_0.8', 'agep+cog9+mriL+mriX', 0.8],
        'nat2': ['nat2_0.8', 'agep+mriL+mriX+csf3+petms3', 0.8],
        'simu': ['simu', 'cli+mriL+mriX', 0.5]
    }
    out_put_name, name, thrd = data_dict[args.data_name]

    data_1 = global_select(name)
    select_data(args.data_root, data_1, out_put_name, thrd)

    if args.data_name == 'simu':
        sample_1100(args.data_root, out_put_name)


def global_select(col_name):
    rawdata = pd.read_csv('./../raw_data/TADPOLE/TADPOLE_D1_D2.csv', low_memory=False, dtype=np.object)

    """
    1. clinical information
    2. MRI: UCSFFSL, UCSFFSX
    3. PET: BAIPETNMRC, UCBERKELEYAV45, UCBERKELEYAV1451
    4. DTI: DTIPOI
    5. CSF: UPENNBIOMK
    """
    base = [0, 8, 2, 54, 10]  # RID, EXAMDATE, VISCODE, DX, DXCHANGE
    col_cli = list(range(11, 18))  # 7
    col_cog = list(range(21, 30))  # 9
    con = base+col_cli+col_cog  # 21
    col_MRI_UCSFFSL = list(range(122, 468))  # 346
    col_MRI_UCSFFSX = list(range(486, 832))  # 346
    col_PET_BAI = list(range(838, 1172))  # 334
    col_PET_AV45 = list(range(1174, 1412))  # 238
    col_PET_AV1451 = list(range(1414, 1656))  # 242
    col_DTI = list(range(1667, 1895))  # 241
    col_CSF = list(range(1902, 1905))  # 3
    col_PETms = list(range(18, 21))  # 3
    col_MRIms = list(range(47, 54))  # 7
    use_agep = ['AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4']
    col_agep = [rawdata.columns.to_list().index(i) for i in rawdata.columns.to_list() if i in use_agep]
    use_agp = ['AGE', 'PTGENDER',  'APOE4']
    col_agp = [rawdata.columns.to_list().index(i) for i in rawdata.columns.to_list() if i in use_agp]

    list_column = {
        'agep+mriL+mriX+csf3+petms3': base + col_agep + col_MRI_UCSFFSL + col_MRI_UCSFFSX + col_CSF + col_PETms,
        'agep+cog9+mriL+mriX': base + col_agep + col_cog + col_MRI_UCSFFSL + col_MRI_UCSFFSX,
        'cli+mriL+mriX': base + col_cli + col_MRI_UCSFFSL + col_MRI_UCSFFSX
    }

    print('col_name: ' + str(col_name))
    data_1 = rawdata.iloc[:, list_column[col_name]]
    data_1 = extract_data(data_1)

    return data_1


def extract_data(data_1):
    dx = data_1[['EXAMDATE', 'RID']]
    data_1 = data_1.drop(['EXAMDATE', 'RID'], axis=1)
    data_1 = pd.concat([dx, data_1], axis=1)

    # sorted by RID and VISCODE
    data_1['RID'] = data_1['RID'].values.astype(int)
    data_1.sort_values(by=['RID', 'VISCODE'], axis=0, ascending=[True, True], inplace=True)

    # replace missing values as np.nan
    data_1.replace(' ', np.nan, inplace=True)
    data_1.replace('', np.nan, inplace=True)
    data_1.replace('-4', np.nan, inplace=True)
    data_1.replace('-1', np.nan, inplace=True)
    data_1.replace('<8', 8, inplace=True)
    data_1.replace('<80', 80, inplace=True)
    data_1.replace('<200', 200, inplace=True)
    data_1.replace('>120', 120, inplace=True)
    data_1.replace('>1300', 1300, inplace=True)

    return data_1


def change_age(data_2):
    # change the baseline age to the age of the visit
    data_2 = data_2.copy()
    for i in range(len(data_2)):
        if pd.isnull(data_2['AGE'][i]):
            continue
        age_bl = float(data_2['AGE'][i])
        vis = data_2['VISCODE'][i]
        if vis == 'bl':
            age = age_bl
        else:
            age = int(vis[1::]) / 12 + age_bl
        data_2.loc[i, 'AGE'] = age

    return data_2


def set_label3(data_3):
    # 1 = Stable:NL to NL, 2 = Stable:MCI to MCI, 3 = Stable:AD to AD,
    # 4 = Conv:NL to MCI, 5 = Conv:MCI to AD, 6 = Conv:NL to AD,
    # 7 = Rev:MCI to NL, 8 = Rev:AD to MCI
    data_4 = data_3[(data_3['DXCHANGE'] == '1') | (data_3['DXCHANGE'] == '2') | (data_3['DXCHANGE'] == '3')]
    data_4.reset_index(drop=True, inplace=True)
    DCdict = {'1': 0, '2': 1, '3': 2}
    data_4 = data_4.copy()
    data_4['label'] = data_4['DXCHANGE'].apply(lambda n: DCdict[n])
    print('\tNC: %d, \tMCI: %d, \tAD: %d' %
        (len(np.where(data_4['DXCHANGE'] == '1')[0]),
        len(np.where(data_4['DXCHANGE'] == '2')[0]),
        len(np.where(data_4['DXCHANGE'] == '3')[0])))
    return data_4


def convert2value(data_4):
    # Convert discrete features (str) into numbers (int)
    # sexdict = {'Male': 0, 'Female': 1}
    # data_3['PTGENDER'] = data_3['PTGENDER'].apply(lambda n: sexdict[n])
    data_5 = data_4.copy()
    if 'PTGENDER' in data_5.columns:
        data_5['PTGENDER'] = pd.factorize(data_4['PTGENDER'])[0].astype(np.uint16)
    if 'PTETHCAT' in data_5.columns:
        data_5['PTETHCAT'] = pd.factorize(data_4['PTETHCAT'])[0].astype(np.uint16)
    if 'PTRACCAT' in data_5.columns:
        data_5['PTRACCAT'] = pd.factorize(data_4['PTRACCAT'])[0].astype(np.uint16)
    if 'PTMARRY' in data_5.columns:
        data_5['PTMARRY'] = pd.factorize(data_4['PTMARRY'])[0].astype(np.uint16)

    features = data_5.iloc[:, 5:-1].values.astype(np.float)
    data_5 = pd.concat([pd.DataFrame(features), data_5.iloc[:, -1]], axis=1)
    return data_5


def select_data(root_dir, data_1, out_name, thrd=0.5):
    if out_name == 'simu':
        head = 'simu'
    else:
        head = 'nat'

    if out_name == 'simu':
        # delete missing values
        col_num = data_1.shape[1]  # column
        data_1 = data_1.dropna(thresh=thrd * col_num, axis=0, inplace=False)  # drop row with non-missing ratio < 50%
        row_num = data_1.shape[0]  # row
        data_1 = data_1.dropna(thresh=thrd * row_num, axis=1, inplace=False)  # drop column with non-missing ratio < 50%
        data_2 = data_1.dropna(axis=0, how='any')  # Delete rows with missing values
    elif thrd == 0.8:
        data_1 = data_1.dropna(subset=['DX'])
        data_1 = data_1.dropna(axis=0, how='all')
        data_1 = data_1.dropna(axis=1, how='all')
        col_num = data_1.shape[1]  # column
        data_2 = data_1.dropna(thresh=thrd * col_num, axis=0, inplace=False)  # drop row with non-missing ratio < thrd
    else:
        raise ValueError(f'wrong out_name： {out_name}')

    data_2.reset_index(drop=True, inplace=True)
    data_3 = change_age(data_2)
    print('before set label: (%d, %d)' % (data_3.shape[0], data_3.shape[1]))
    # set label in the last column
    data_4 = set_label3(data_3)
    print('after set label: (%d, %d)' % (data_4.shape[0], data_4.shape[1]))

    os.makedirs(os.path.join(root_dir, head), exist_ok=True)
    data_4.to_csv(os.path.join(root_dir, head, f'{out_name}.csv'), index=False)
    print('Save!')

    data_5 = convert2value(data_4)
    data_5.reset_index(drop=False, inplace=True)
    data_5.to_csv(os.path.join(root_dir, head, f'{out_name}_value.csv'), index=False, header=False)
    print('Save value!')  # column1: index, column2: RIS, column-1: label


def sample_1100(root_dir, data_name):
    head = 'simu'
    lost_kind = 50
    set_num = lost_kind * 22
    data_full = pd.read_csv(os.path.join(root_dir, head, f'{data_name}_value.csv'), header=None)
    data_0 = data_full.iloc[np.where(data_full.iloc[:, -1] == 0)].sample(n=350, random_state=2029, axis=0)
    data_1 = data_full.iloc[np.where(data_full.iloc[:, -1] == 1)].sample(n=500, random_state=2029, axis=0)
    data_2 = data_full.iloc[np.where(data_full.iloc[:, -1] == 2)].sample(n=250, random_state=2029, axis=0)
    data_part0 = []
    for i in range(lost_kind):
        data_part0.extend(data_0.iloc[i * 7:(i + 1) * 7, :].values)
        data_part0.extend(data_1.iloc[i * 10:(i + 1) * 10, :].values)
        data_part0.extend(data_2.iloc[i * 5:(i + 1) * 5, :].values)
    data_part0 = pd.DataFrame(data_part0)
    os.makedirs(os.path.join(root_dir, head), exist_ok=True)
    data_part0.to_csv(os.path.join(root_dir, head, f'{data_name}_{set_num}.csv'), index=False, header=False)
    print('finish sample 1100 from 1139')


if __name__ == '__main__':
    main()

