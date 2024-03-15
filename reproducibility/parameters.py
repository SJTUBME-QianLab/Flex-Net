import pandas as pd
import numpy as np


def get_rand():
    rand = dict()
    parameters = pd.read_excel('./parameters.xlsx', sheet_name=None)
    for key, value in parameters.items():
        if key in ['nat1', 'nat2', 'BC', 'CC', 'CK', 'HC', 'HD', 'HP', 'HS', 'PI']:
            rand_list = np.unique(value['rand'])
            assert len(rand_list) == 10
            rand[key] = rand_list.astype(int)
        elif key in ['MCAR', 'MAR', 'MNAR']:
            if key not in rand:
                rand[key] = dict()
            for lost_name in np.unique(value['lost_num']):
                sub = value[value['lost_num'] == lost_name]
                rand_list = np.unique(sub['rand'])
                assert len(rand_list) == 10
                rand[key][lost_name] = rand_list.astype(int)
        else:
            pass
    return rand


def get_seed_RF():
    rand = dict()
    parameters = pd.read_excel('./parameters.xlsx', sheet_name=None)
    for key, value in parameters.items():
        if key in ['nat1', 'nat2', 'BC', 'CC', 'CK', 'HC', 'HD', 'HP', 'HS', 'PI']:
            if key not in rand:
                rand[key] = dict()
            for method in np.unique(value['method']):
                if 'RF' in method:
                    sub = value[value['method'] == method]
                    seed_list = sub['para_name0']
                    assert len(seed_list) == 10
                    # if len(seed_list) != 10:
                    #     print(key, method, seed_list)
                    #     raise ValueError
                    if set(seed_list) == {'default'}:
                        seed_list = [2021] * 10
                    else:
                        seed_list = seed_list.astype(int)
                    rand[key][method] = seed_list

        elif key in ['MCAR', 'MAR', 'MNAR']:
            if key not in rand:
                rand[key] = dict()
            for lost_name in np.unique(value['lost_num']):
                if lost_name not in rand[key]:
                    rand[key][lost_name] = dict()
                sub = value[value['lost_num'] == lost_name]
                for method in np.unique(sub['method']):
                    if 'RF' in method:
                        sub1 = sub[sub['method'] == method]
                        seed_list = sub1['para_name0']
                        assert len(seed_list) == 10
                        # if len(seed_list) != 10:
                        #     print(key, lost_name, method, seed_list)
                        #     raise ValueError
                        if set(seed_list) == {'default'}:
                            seed_list = [2021] * 10
                        else:
                            seed_list = seed_list.astype(int)
                        rand[key][lost_name][method] = seed_list
        else:
            pass
    return rand


def get_seed(method):
    rand = dict()
    parameters = pd.read_excel('./parameters.xlsx', sheet_name=None)
    for key, value in parameters.items():
        if key in ['nat1', 'nat2', 'BC', 'CC', 'CK', 'HC', 'HD', 'HP', 'HS', 'PI']:
            sub = value[value['method'] == method]
            if len(sub) == 0:
                continue
            seed_list = sub['seed'].astype(int)
            # assert len(seed_list) == 10
            if len(seed_list) != 10:
                print(key, method, seed_list)
                raise ValueError
            rand[key] = seed_list

        elif key in ['MCAR', 'MAR', 'MNAR']:
            if key not in rand:
                rand[key] = dict()
            for lost_name in np.unique(value['lost_num']):
                if lost_name not in rand[key]:
                    rand[key][lost_name] = dict()
                sub = value[value['lost_num'] == lost_name]
                sub1 = sub[sub['method'] == method]
                if len(sub1) == 0:
                    continue
                seed_list = sub1['seed'].astype(int)
                # assert len(seed_list) == 10
                if len(seed_list) != 10:
                    print(key, lost_name, method, seed_list)
                    raise ValueError
                rand[key][lost_name] = seed_list
        else:
            pass
    return rand


def get_params(method):
    assert method in ['Ours', 'GCN']
    params = dict()
    parameters = pd.read_excel('./parameters.xlsx', sheet_name=method, index_col=0)
    end_loc = parameters.columns.get_loc('para_name0') + 1
    for data_name in parameters.index:
        params[data_name] = parameters.loc[data_name, :][:end_loc].to_dict()
    return params


rand_list_dict = get_rand()
seed_RF = get_seed_RF()
params_ours = get_params('Ours')
params_GCN = get_params('GCN')
