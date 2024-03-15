import numpy as np
import pandas as pd
import os
import time
import pickle
import scipy
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcdefaults()
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 900
plt.rcParams['figure.dpi'] = 900
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE


def main():
    # # Fig. 3, a-d
    # plot_nat1_imp()
    # plot_nat1_drop()
    # plot_nat2_imp()
    # plot_nat2_drop()

    # Extended Data Fig. 3, a-b
    plot_MissPatternModal_nat1()
    plot_MissPatternModal_nat2()

    # # Extended Data Fig. 4, a-d
    # plot_ScatterLossLabel_nat1()
    # plot_MissPatternRatio_nat1()
    # plot_ScatterLossLabel_nat2()
    # plot_MissPatternRatio_nat2()


def load_data(name):
    df = pd.read_excel('./all_results.xlsx', sheet_name=name)
    df = df[['method', 'acc']]

    colors1 = sns.color_palette('Set3')
    colors2 = sns.color_palette('tab10')
    colorsSet = [colors1[0]] + [colors1[2]] + colors1[4:12] + colors2[-2:] + [colors1[3]]

    fill_methods = {
        'mean': 'Mean', 'knn.3': 'KNN', 'softimpute': 'SoftImpute', 'missForest': 'MissForest'
    }
    colors = dict()
    dfR = []
    for tt in ['RF', 'SVM', 'GCN']:
        for i, kk in enumerate(list(fill_methods.keys())):
            colors[f'{fill_methods[kk]}+{tt}'] = colorsSet[i]
            dfi = df[df['method'] == f'{kk}+{tt}']
            dfi.loc[:, 'method'] = f'{fill_methods[kk]}+{tt}'
            dfR.append(dfi)
    colors['Ours (Flex-Net)'] = colorsSet[-1]
    dfi = df[df['method'] == 'Ours']
    dfi.loc[:, 'method'] = 'Ours (Flex-Net)'
    dfR.append(dfi)

    dfR = pd.concat(dfR, axis=0)
    return dfR, colors


def load_data_Drop(name):
    df = pd.read_excel('./all_results.xlsx', sheet_name=name)
    df = df[['method', 'acc']]

    colors1 = sns.color_palette('Set3')
    colors2 = sns.color_palette('Set2')
    colorsSet = colors2[:2] + [colors2[3]] + colors2[5:7] + [colors1[3]]

    clf_methods = ['RF', 'SVM', 'GCN', 'GCN+risk', 'AutoMetric']
    colors = dict()
    dfR = []
    for i, t0 in enumerate(clf_methods):
        if t0 == 'GCN+risk':
            tt = 'GCNRisk'
        else:
            tt = t0
        if name == 'nat1':
            colors[f'DS+{tt}'] = colorsSet[2]
            dfi = df[df['method'] == f'dS+{t0}']
            dfi.loc[:, 'method'] = f'DS+{tt}'
            dfR.append(dfi)
        colors[f'DA+{tt}'] = colorsSet[4]
        dfi = df[df['method'] == f'dF+{t0}']
        dfi.loc[:, 'method'] = f'DA+{tt}'
        dfR.append(dfi)

    colors['Ours (Flex-Net)'] = colorsSet[-1]
    dfi = df[df['method'] == 'Ours']
    dfi.loc[:, 'method'] = 'Ours (Flex-Net)'
    dfR.append(dfi)
    dfR = pd.concat(dfR, axis=0)

    return dfR, colors


def plot_nat1_imp():
    df, colors = load_data('nat1')
    fig = plt.figure(figsize=(2, 5))

    ax = plt.subplot(1, 1, 1)

    configs = {'width': 0.8, 'flierprops': {'marker': '.'},}
    sns.boxplot(
        ax=ax, data=df, x='acc', y='method', orient='h',
        palette=colors, **configs,
    )

    ax.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    ax.tick_params(
        axis='y', length=4, width=1.5,
    )
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)

    ax.set_facecolor('None')  # 透明背景
    ax.grid(axis='y', lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(len(colors)-0.3, -0.7)
    print(ax.get_xlim())
    ax.set_xlim(0.86, 0.915)
    ax.set_xticks([0.87, 0.89, 0.91])
    ax.set_ylabel('')
    ax.set_xlabel('Accuracy', fontsize=12)
            
    for label in ax.get_yticklabels():
        if label.get_text() == 'Ours (Flex-Net)':
            label.set_color('firebrick')

    plt.savefig('./figures/Fig3_nat1_Imp.svg', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_nat1_drop():
    df, colors = load_data_Drop('nat1')
    fig = plt.figure(figsize=(2, 5/13*11))
    ax = plt.subplot(1, 1, 1)

    configs = {'width': 0.8, 'flierprops': {'marker': '.'},}
    sns.boxplot(
        ax=ax, data=df, x='acc', y='method', orient='h',
        palette=colors, **configs, 
    )

    ax.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    ax.tick_params(
        axis='y', length=4, width=1.5,
    )
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)

    ax.set_facecolor('None')  # 透明背景
    ax.grid(axis='y', lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(len(colors)-0.3, -0.7)
    print(ax.get_xlim())
    ax.set_xlim(0.835, 0.914)
    ax.set_xticks([0.84, 0.87, 0.9])
    ax.set_ylabel('')
    ax.set_xlabel('Accuracy', fontsize=12)
            
    for label in ax.get_yticklabels():
        if label.get_text() == 'Ours (Flex-Net)':
            label.set_color('firebrick')

    plt.savefig('./figures/Fig3_nat1_Drop.svg', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_nat2_imp():
    df, colors = load_data('nat2')
    fig = plt.figure(figsize=(2, 5))

    ax = plt.subplot(1, 1, 1)

    configs = {'width': 0.8, 'flierprops': {'marker': '.'},}
    sns.boxplot(
        ax=ax, data=df, x='acc', y='method', orient='h',
        palette=colors, **configs,
    )
    ax.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    ax.tick_params(
        axis='y', labelsize=12, length=4, width=1.5,
    )
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)

    ax.set_facecolor('None')  # 透明背景
    ax.grid(axis='y', lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(len(colors)-0.3, -0.7)
    print(ax.get_xlim())
    ax.set_xlim(0.65, 0.82)
    ax.set_xticks([0.7, 0.75, 0.8])
    ax.set_ylabel('')
    ax.set_xlabel('Accuracy', fontsize=12)

    for label in ax.get_yticklabels():
        if label.get_text() == 'Ours (Flex-Net)':
            label.set_color('firebrick')

    plt.savefig('./figures/Fig3_nat2_Imp.svg', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_nat2_drop():
    df, colors = load_data_Drop('nat2')
    fig = plt.figure(figsize=(2, 5/13*6))
    ax = plt.subplot(1, 1, 1)

    configs = {'width': 0.8, 'flierprops': {'marker': '.'},}
    sns.boxplot(
        ax=ax, data=df, x='acc', y='method', orient='h',
        palette=colors, **configs, 
    )

    ax.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    ax.tick_params(
        axis='y', length=4, width=1.5,
    )
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)

    ax.set_facecolor('None')  # 透明背景
    ax.grid(axis='y', lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(len(colors)-0.3, -0.7)
    print(ax.get_xlim())
    ax.set_xlim(0.65, 0.82)
    ax.set_xticks([0.7, 0.75, 0.8])
    ax.set_ylabel('')
    ax.set_xlabel('Accuracy', fontsize=12)
            
    for label in ax.get_yticklabels():
        if label.get_text() == 'Ours (Flex-Net)':
            label.set_color('firebrick')

    plt.savefig('./figures/Fig3_nat2_Drop.svg', bbox_inches='tight', pad_inches=0)
    plt.show()


def load_modal(name):
    data = pd.read_csv(f'./../data/nat/{name}_0.8.csv')
    col_names = data.columns[5:-1]
    lossYN = pd.read_excel(f'./../data/nat/{name}_0.8_/info_count.xlsx', sheet_name='part_count')
    lossTF = pd.DataFrame(columns=col_names, index=[int(kk) for kk in lossYN['new_loss_flag'][:-1]])
    for i in range(len(lossYN) - 1):
        lossTF.iloc[i, :] = np.array([kk=='Y' for kk in lossYN.loc[i, 'loss'][1:-1]])
    return lossTF


def plot_MissPatternModal_nat1():
    df = load_modal('nat1')
    data = df.iloc[:, np.where(df.sum(axis=0) > 0)[0]].astype(int).T
    print(data.shape)
    cmap = plt.get_cmap('PuBu', 15)
    colorsSet = [cmap(kk) for kk in [5, 8]]

    fig = plt.figure(figsize=(7.2/36*31/data.shape[0]*data.shape[1], 7.2/36*31), facecolor=None)
    ax1 = fig.add_subplot(1, 1, 1)
    a1 = sns.heatmap(
        ax=ax1, data=data,
        cmap=colorsSet,
        linewidth=1., linecolor='white',
        cbar=None,
        square=False,
    )
    ax1.tick_params(
        axis='both', labelsize=10, 
        length=0, #width=1.5,
        bottom=False, top=False, left=False, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    ax1.set_xticks(np.arange(data.shape[1]) + 0.5, 
                   [str(kk) for kk in np.arange(1, 1+data.shape[1])], rotation=0)
    ax1.set_xlabel('Missing pattern', fontsize=14)
    ax1.set_ylabel('Attribute', fontsize=14)

    plt.savefig('./figures/ExFig3_nat1_Modal.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def plot_MissPatternModal_nat2():
    df = load_modal('nat2')
    data = df.iloc[:, np.where(df.sum(axis=0) > 0)[0]].astype(int).T
    print(data.shape)
    cmap = plt.get_cmap('PuBu', 15)
    colorsSet = [cmap(kk) for kk in [5, 8]]

    fig = plt.figure(figsize=(7.2/data.shape[0]*data.shape[1], 7.2), facecolor=None)
    ax1 = fig.add_subplot(1, 1, 1)

    a1 = sns.heatmap(
        ax=ax1, data=data, 
        cmap=colorsSet,
        linewidth=1., linecolor='white',
        cbar=None,
        square=False,
    )
    ax1.tick_params(
        axis='both', labelsize=10, 
        length=0,
        bottom=False, top=False, left=False, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    ax1.set_xticks(np.arange(4, data.shape[1], 5) + 0.5, 
                   [str(kk) for kk in np.arange(5, 1+data.shape[1], 5)], rotation=0)
    ax1.set_xlabel('Missing pattern', fontsize=14)
    ax1.set_ylabel('Attribute', fontsize=14)

    plt.savefig('./figures/ExFig3_nat2_Modal.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def get_heatmap(name):
    label_dict = {0: 'NC', 1: 'MCI', 2: 'AD'}
    data_path = f'../data/nat/{name}_0.8_/divide_2'
    train = pd.read_csv(os.path.join(data_path, 'index_data_label_lost_0_train.csv'), header=None)
    test = pd.read_csv(os.path.join(data_path, 'index_data_label_lost_0_test.csv'), header=None)
    data = pd.concat([train, test], axis=0).reset_index(drop=True, inplace=False)
    features = data.iloc[:, 1:-2]
    label = data.iloc[:, -2]
    loss_flag = data.iloc[:, -1]
    del train, test, data

    features = np.isnan(features).astype(int)
    pipeline = Pipeline([
        ('scaling', StandardScaler()),  # default along axis=0, so the matrix should be samples*features
        ('pca', PCA(n_components=1, random_state=2023))
    ])
    newX = pipeline.fit_transform(features)

    embs = pd.DataFrame({
        'Label': [label_dict[kk] for kk in label],
        'Missing pattern': loss_flag,
        'x': newX[:, 0],
    })
    embs['x'] = ['%.6f' % kk for kk in embs['x']]
    counts = embs.groupby(by=['Label', 'Missing pattern']).count().reset_index(drop=False)
    counts.rename(columns={'x': 'count'}, inplace=True)
    loss_loc = embs.drop_duplicates(keep='first', ignore_index=True)
    assert set(embs['Missing pattern']) == set(loss_loc['Missing pattern'])
    loc_dict = sorted(set([float(kk) for kk in embs['x']]))
    loc_dict = dict(zip(['%.6f' % kk for kk in loc_dict], range(len(loc_dict))))
    counts = pd.merge(counts, loss_loc, on=['Label', 'Missing pattern'])
    counts['int'] = [loc_dict[kk] for kk in counts['x']]
    heatmaps = []
    for kk in ['NC', 'MCI', 'AD']:
        dfi = counts[counts['Label'] == kk][['count', 'Missing pattern']]
        dfi = pd.DataFrame(dfi['count'].values, columns=[kk], index=dfi['Missing pattern'].values)
        heatmaps.append(dfi)
    heatmaps = pd.concat(heatmaps, axis=1)
    heatmaps.sort_index(inplace=True)
    return heatmaps


def get_tsne(name):
    label_dict = {0: 'NC', 1: 'MCI', 2: 'AD'}
    data_path = f'../data/nat/{name}_0.8_/divide_2'
    train = pd.read_csv(os.path.join(data_path, 'index_data_label_lost_0_train.csv'), header=None)
    test = pd.read_csv(os.path.join(data_path, 'index_data_label_lost_0_test.csv'), header=None)
    data = pd.concat([train, test], axis=0).reset_index(drop=True, inplace=False)
    features = data.iloc[:, 1:-2]
    label = data.iloc[:, -2]
    loss_flag = data.iloc[:, -1]
    del train, test, data

    features = features.iloc[:, np.where(features.isna().sum(axis=0) == 0)[0]]  # complete attributes
    pipeline = Pipeline([
        ('scaling', StandardScaler()),  # default along axis=0, so the matrix should be samples*features
        ('tsne', TSNE(n_components=2, init='pca', learning_rate='auto', random_state=2023))
    ])
    newX = pipeline.fit_transform(features)

    return pd.DataFrame({
        'Label': [label_dict[kk] for kk in label],
        'Missing pattern': [kk + 1 for kk in loss_flag],
        'x': newX[:, 0],
        'y': newX[:, 1],
    })


def plot_MissPatternRatio_nat1():
    df = get_heatmap('nat1')
    for i in range(len(df)):
        df.iloc[i, :] = df.iloc[i, :] / sum(df.iloc[i, :])
    
    fig = plt.figure(figsize=(5, 1.3), facecolor=None)
    gs = fig.add_gridspec(
        nrows=1, ncols=2,  width_ratios=(50, 1),
        wspace=0.05, 
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    
    a1 = sns.heatmap(
        ax=ax1, data=df.T, 
        cmap=plt.cm.PuBu, # 'crest',
        annot=True, fmt='.2f', annot_kws={'fontsize': 10},
        linewidth=.5, linecolor='white',
        cbar_ax=ax_cbar, cbar_kws={'orientation': 'vertical'},
    )
    ax1.tick_params(
        axis='both', labelsize=12, 
        length=0,
        bottom=False, top=False, left=False, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    ax_cbar.tick_params(
        axis='y', labelsize=10, 
        length=2, width=1.5,
        right=True, labelright=True,
    )
    ax1.set_xticks(np.arange(len(df)) + 0.5, 
                   [str(kk) for kk in np.arange(1, 1+len(df))], rotation=0)
    ax1.set_yticks(np.arange(3) + 0.5, 
                   ['NC', 'MCI', 'AD'], rotation=0)
    ax1.set_xlabel('Missing pattern', fontsize=14)
    ax1.set_ylabel('Class', fontsize=14)
    
    plt.savefig('./figures/ExFig4_nat1_heatmap.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def plot_ScatterLossLabel_nat1():
    df = get_tsne('nat1')
    markers = {'NC': 'o', 'MCI': 's', 'AD': 'X'}
    color_num = len(set(df['Missing pattern']))  # 9

    cmap = plt.get_cmap('Spectral', 12)
    colorsSet = [cmap(kk) for kk in list(range(4)) + list(range(7, 12))]

    # (-92.09279365539551, 97.38758583068848) (-79.45434799194337, 84.6346305847168)
    scopex = -93, 98  # 
    scopey = -80, 85  # 

    fig = plt.figure(figsize=(3.4*4, 3.4/(scopex[1] - scopex[0])*(scopey[1] - scopey[0])), facecolor='none')
    gs = fig.add_gridspec(
        nrows=1, ncols=4, width_ratios=(1, 1, 1, 1),
        wspace=0.04,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])

    sns.scatterplot(
        ax=ax0, data=df, x='x', y='y', 
        style='Label', style_order=markers.keys(), markers=markers,
        hue='Missing pattern', hue_order=range(1, 1+color_num),
        palette=colorsSet, s=16, alpha=1, 
        legend='brief',
    )
    print(ax0.get_xlim(), ax0.get_ylim())
    
    handles, labels = ax0.get_legend_handles_labels()
    color_hl = handles[1:(color_num+1)], labels[1:(color_num+1)]
    style_hl = handles[(color_num+2):], labels[(color_num+2):]    

    color_leg = ax0.legend(
        *color_hl,
        title='Missing pattern', 
        loc='upper right', bbox_to_anchor=(-0.01, 1), 
        borderaxespad=0.2, 
        fontsize=10, labelspacing=0.4,
        ncol=3, handlelength=0.8, columnspacing=1.35, handletextpad=0.5, borderpad=0.5,
    )
    style_leg = ax0.legend(
        *style_hl,
        title='Class', 
        loc='upper right', bbox_to_anchor=(-0.54, 1), 
        borderaxespad=0.2, 
        fontsize=10, labelspacing=0.4,
        ncol=1, handlelength=0.8, columnspacing=0.5, handletextpad=0.5, borderpad=0.5,
    )
    ax0.add_artist(color_leg)

    for axi, kk in zip([ax1, ax2, ax3], ['NC', 'MCI', 'AD']):
        sns.scatterplot(
            ax=axi, data=df[df['Label']!=kk], x='x', y='y', 
            style='Label', style_order=markers.keys(), markers=markers,
            color='white', edgecolor='grey', linewidth=0.5, s=16, alpha=0.1,
            legend=None,
        )
        sns.scatterplot(
            ax=axi, data=df[df['Label']==kk], x='x', y='y', 
            style='Label', style_order=markers.keys(), markers=markers,
            hue='Missing pattern', hue_order=range(1, 1+color_num),
            palette=colorsSet, s=16, alpha=1, 
            legend=None,
        )
        axi.text(scopex[1]-18, scopey[1]-12, kk, 
                 fontsize=12, horizontalalignment='center', verticalalignment='center')
    ax0.text(scopex[1]-18, scopey[1]-12, 'All', 
             fontsize=12, horizontalalignment='center', verticalalignment='center')

    for axi in [ax0, ax1, ax2, ax3]:
        axi.set_xlim(*scopex)
        axi.set_ylim(*scopey)
        axi.tick_params(
            axis='both', labelsize=12, length=2, 
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labeltop=False, labelleft=False, labelright=False,
        )
        axi.set_xlabel('')
        axi.set_ylabel('')
        axi.spines['left'].set_linewidth(1.5)
        axi.spines['right'].set_linewidth(1.5)
        axi.spines['bottom'].set_linewidth(1.5)
        axi.spines['top'].set_linewidth(1.5)
        
    plt.savefig('./figures/ExFig4_nat1_scatter.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def plot_MissPatternRatio_nat2():
    df = get_heatmap('nat2')
    for i in range(len(df)):
        df.iloc[i, :] = df.iloc[i, :] / sum(df.iloc[i, :])

    fig = plt.figure(figsize=(16, 1.4), facecolor=None)
    gs = fig.add_gridspec(
        nrows=1, ncols=2, width_ratios=(160, 1),
        wspace=0.02,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])

    a1 = sns.heatmap(
        ax=ax1, data=df.T,
        cmap=plt.cm.PuBu,  # 'crest',
        annot=True, fmt='.2f', annot_kws={'fontsize': 10},
        linewidth=.5, linecolor='white',
        cbar_ax=ax_cbar, cbar_kws={'orientation': 'vertical'},
    )
    ax1.tick_params(
        axis='both', labelsize=12,
        length=-0.5,  # width=1.5,
        bottom=False, top=False, left=False, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    ax_cbar.tick_params(
        axis='y', labelsize=10,
        length=2, width=1.5,
        right=True, labelright=True,
    )
    ax1.set_xticks(np.arange(len(df)) + 0.5,
                   [str(kk) for kk in np.arange(1, 1 + len(df))], rotation=0)
    ax1.set_yticks(np.arange(3) + 0.5,
                   ['NC', 'MCI', 'AD'], rotation=0)
    ax1.set_xlabel('Missing pattern', fontsize=14)
    ax1.set_ylabel('Class', fontsize=14)

    plt.savefig(f'./figures/ExFig4_nat2_heatmap.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def plot_ScatterLossLabel_nat2():
    df = get_tsne('nat2')
    markers = {'NC': 'o', 'MCI': 's', 'AD': 'X'}
    color_num = len(set(df['Missing pattern']))  # 9

    cmap = plt.get_cmap('Spectral', 43)
    colorsSet = [cmap(kk) for kk in list(range(16)) + list(range(26, 43))]

    # (-94.17243728637695, 98.6556526184082) (-82.87330360412598, 87.80539436340332)
    scopex = -95, 99  #
    scopey = -83, 88  #

    fig = plt.figure(figsize=(3.4 * 4, 3.4 / (scopex[1] - scopex[0]) * (scopey[1] - scopey[0])), facecolor='none')
    gs = fig.add_gridspec(
        nrows=1, ncols=4, width_ratios=(1, 1, 1, 1),
        wspace=0.04,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])

    sns.scatterplot(
        ax=ax0, data=df, x='x', y='y',
        style='Label', style_order=markers.keys(), markers=markers,
        hue='Missing pattern', hue_order=range(1, 1 + color_num),
        palette=colorsSet, s=16, alpha=1,
        legend='brief',
    )
    print(ax0.get_xlim(), ax0.get_ylim())

    handles, labels = ax0.get_legend_handles_labels()
    color_hl = handles[1:(color_num + 1)], labels[1:(color_num + 1)]
    style_hl = handles[(color_num + 2):], labels[(color_num + 2):]
    color_leg = ax0.legend(
        *color_hl,
        title='Missing pattern',
        loc='upper right', bbox_to_anchor=(-0.01, 1),
        borderaxespad=0.2,
        fontsize=10, labelspacing=0.4,
        ncol=3, handlelength=0.8, columnspacing=0.5, handletextpad=0.5, borderpad=0.5,
    )
    style_leg = ax0.legend(
        *style_hl,
        title='Class',
        loc='upper right', bbox_to_anchor=(-0.54, 1),
        borderaxespad=0.2,
        fontsize=10, labelspacing=0.4,
        ncol=1, handlelength=0.8, columnspacing=0.5, handletextpad=0.5, borderpad=0.5,
    )
    ax0.add_artist(color_leg)

    for axi, kk in zip([ax1, ax2, ax3], ['NC', 'MCI', 'AD']):
        sns.scatterplot(
            ax=axi, data=df[df['Label'] != kk], x='x', y='y',
            style='Label', style_order=markers.keys(), markers=markers,
            color='white', edgecolor='grey', linewidth=0.5, s=16, alpha=0.1,
            legend=None,
        )
        sns.scatterplot(
            ax=axi, data=df[df['Label'] == kk], x='x', y='y',
            style='Label', style_order=markers.keys(), markers=markers,
            hue='Missing pattern', hue_order=range(1, 1 + color_num),
            palette=colorsSet, s=16, alpha=1,
            legend=None,
        )
        axi.text(scopex[1] - 18, scopey[1] - 12, kk,
                 fontsize=12, horizontalalignment='center', verticalalignment='center')
    ax0.text(scopex[1] - 18, scopey[1] - 12, 'All',
             fontsize=12, horizontalalignment='center', verticalalignment='center')

    for axi in [ax0, ax1, ax2, ax3]:
        axi.set_xlim(*scopex)
        axi.set_ylim(*scopey)
        axi.tick_params(
            axis='both', labelsize=12, length=2,
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labeltop=False, labelleft=False, labelright=False,
        )
        axi.set_xlabel('')
        axi.set_ylabel('')
        axi.spines['left'].set_linewidth(1.5)
        axi.spines['right'].set_linewidth(1.5)
        axi.spines['bottom'].set_linewidth(1.5)
        axi.spines['top'].set_linewidth(1.5)

    plt.savefig('./figures/ExFig4_nat2_scatter.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


if __name__ == '__main__':
    main()
