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


def main():
    # Fig. 5 a (MCAR)
    df, colors = load_data('MCAR')
    plot_MCAR(df, colors, lost='L663p0.05', labelleft=True)
    plot_MCAR(df, colors, lost='L663p0.1', labelleft=False)
    plot_MCAR(df, colors, lost='L663p0.15', labelleft=False)

    df['lost_num'] = [float(kk.split('L663p')[1]) for kk in df['lost_num']]
    plot_line(df.copy(), name='MCAR', err='se', leg=False)

    # Fig. 5 b (MAR)
    df, colors = load_data('MAR')
    for ll in ['L331p0.4', 'L331p0.6', 'L331p0.8', 'L530p0.5', 'L596p0.6', 'L596p0.8']:
        plot_MAR(df, colors, lost=ll, labelleft=False)
    plot_MAR(df, colors, lost='L331p0.2', labelleft=True)

    df = df[df['lost_num'].str.contains('L331')]
    df['lost_num'] = [round(float(kk.split('L331p')[1]) * 0.5, 2) for kk in df['lost_num']]
    df = df[df['lost_num'].isin([0.10, 0.20, 0.30, 0.40])]
    plot_line(df.copy(), name='MAR', err='se', leg=False)

    # Fig. 5 c (MNAR)
    df, colors = load_data('MNAR')
    plot_MNAR(df, colors, lost='L663p0.05', labelleft=True)
    plot_MNAR(df, colors, lost='L663p0.1', labelleft=False)
    plot_MNAR(df, colors, lost='L663p0.15', labelleft=False)

    df['lost_num'] = [float(kk.split('L663p')[1]) for kk in df['lost_num']]
    plot_line(df.copy(), name='MNAR', err='se', leg=True)


def load_data(name):
    df = pd.read_excel('./all_results.xlsx', sheet_name=name)
    df = df[['lost_num', 'method', 'acc']]

    colors1 = sns.color_palette('Set3')
    colors2 = sns.color_palette('tab10')
    colorsSet = [colors1[0]] + [colors1[2]] + colors1[4:12] + [colors2[-2]] + [colors1[3]]

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


def plot_MCAR(df, colors, lost, labelleft):
    xlim_dict = {
        'L663p0.05': [0.740, 0.855, [0.76, 0.80, 0.84]],
        'L663p0.1': [0.73, 0.841, [0.75, 0.79, 0.83]],
        'L663p0.15': [0.707, 0.832, [0.72, 0.77, 0.82]]
    }

    pi = round(float(lost.split('p')[1]), 2)
    title = '$R_a$=$R_s$=100%\n$R_m$={:.0f}%'.format(pi * 100)

    fig = plt.figure(figsize=(2, 5))
    ax = plt.subplot(1, 1, 1)
    df1 = df[df['lost_num'] == lost]

    configs = {'width': 0.8, 'flierprops': {'marker': '.'}, }
    sns.boxplot(
        ax=ax, data=df1, x='acc', y='method', orient='h',
        palette=colors, **configs,
    )

    ax.set_title(title, fontsize=12)
    ax.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=True, top=False, left=labelleft, right=False,
        labelbottom=True, labeltop=False, labelleft=labelleft, labelright=False,
    )
    if labelleft:
        ax.tick_params(axis='y', length=4, width=1.5)
        for label in ax.get_yticklabels():
            if label.get_text() == 'Ours (Flex-Net)':
                label.set_color('firebrick')

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)

    ax.set_facecolor('None')
    ax.grid(axis='y', lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(len(colors) - 0.3, -0.7)
    print(lost, ax.get_xlim())
    ax.set_xlim(xlim_dict[lost][0], xlim_dict[lost][1])
    ax.set_xticks(xlim_dict[lost][2])
    ax.set_ylabel('')
    if lost == 'L663p0.1':
        ax.set_xlabel('Accuracy', fontsize=12)
    else:
        ax.set_xlabel('')

    plt.savefig(f'./figures/simu/MCAR.{lost}.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def plot_MAR(df, colors, lost, labelleft):
    xlim_dict = {
        'L331p0.2': [0.725, 0.86, [0.74, 0.79, 0.84]],
        'L331p0.4': [0.70, 0.865, [0.72, 0.78, 0.84]],
        'L331p0.6': [0.66, 0.86, [0.68, 0.76, 0.84]],
        'L331p0.8': [0.61, 0.865, [0.64, 0.74, 0.84]],
        'L530p0.5': [0.64, 0.805, [0.66, 0.72, 0.78]],
        'L596p0.6': [0.535, 0.74, [0.56, 0.64, 0.72]],
        'L596p0.8': [0.505, 0.71, [0.53, 0.61, 0.69]],
    }

    pc = round(int(lost.split('L')[1].split('p')[0]) / 663, 1)
    pi = round(float(lost.split('p')[1]), 1)
    pm = pc * pi
    title = '$R_s$=100%, $p_s$={:.0f}%\n$R_a$={:.0f}%, $R_m$={:.0f}%'.format(pi * 100, pc * 100, pm * 100)

    fig = plt.figure(figsize=(2, 5))
    ax = plt.subplot(1, 1, 1)
    df1 = df[df['lost_num'] == lost]

    configs = {'width': 0.8, 'flierprops': {'marker': '.'}, }
    sns.boxplot(
        ax=ax, data=df1, x='acc', y='method', orient='h',
        palette=colors, **configs,
    )

    ax.set_title(title, fontsize=12)
    ax.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=True, top=False, left=labelleft, right=False,
        labelbottom=True, labeltop=False, labelleft=labelleft, labelright=False,
    )
    if labelleft:
        ax.tick_params(axis='y', length=4, width=1.5)
        for label in ax.get_yticklabels():
            if label.get_text() == 'Ours (Flex-Net)':
                label.set_color('firebrick')

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)

    ax.set_facecolor('None')
    ax.grid(axis='y', lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(len(colors) - 0.3, -0.7)
    print(lost, ax.get_xlim())
    ax.set_xlim(xlim_dict[lost][0], xlim_dict[lost][1])
    ax.set_xticks(xlim_dict[lost][2])
    ax.set_ylabel('')
    if lost == 'L331p0.8':
        ax.set_xlabel('Accuracy', fontsize=12)
    else:
        ax.set_xlabel('')

    plt.savefig(f'./figures/simu/MAR.{lost}.svg', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_MNAR(df, colors, lost, labelleft):
    xlim_dict = {
        'L663p0.05': [0.745, 0.86, [0.76, 0.80, 0.84]],
        'L663p0.1': [0.725, 0.86, [0.74, 0.79, 0.84]],
        'L663p0.15': [0.70, 0.83, [0.72, 0.76, 0.8]]
    }

    pi = round(float(lost.split('p')[1]), 2)
    title = '$R_a$=100%\n$R_s$=$R_m$={:.0f}%'.format(pi * 100)

    fig = plt.figure(figsize=(2, 5))
    ax = plt.subplot(1, 1, 1)
    df1 = df[df['lost_num'] == lost]

    configs = {'width': 0.8, 'flierprops': {'marker': '.'}, }
    sns.boxplot(
        ax=ax, data=df1, x='acc', y='method', orient='h',
        palette=colors, **configs,
    )

    ax.set_title(title, fontsize=12)
    ax.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=True, top=False, left=labelleft, right=False,
        labelbottom=True, labeltop=False, labelleft=labelleft, labelright=False,
    )
    if labelleft:
        ax.tick_params(axis='y', length=4, width=1.5)
        for label in ax.get_yticklabels():
            if label.get_text() == 'Ours (Flex-Net)':  # 假设我们要加粗的tick label是'3'
                #             label.set_weight('bold')
                label.set_color('firebrick')

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)

    ax.set_facecolor('None')
    ax.grid(axis='y', lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(len(colors) - 0.3, -0.7)
    print(lost, ax.get_xlim())
    ax.set_xlim(xlim_dict[lost][0], xlim_dict[lost][1])
    ax.set_xticks(xlim_dict[lost][2])
    ax.set_ylabel('')
    if lost == 'L663p0.1':
        ax.set_xlabel('Accuracy', fontsize=12)
    else:
        ax.set_xlabel('')

    plt.savefig(f'./figures/simu/MNAR.{lost}.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def plot_line(df, name, err=None, leg=True):
    df['lost_num'] = df['lost_num'] * 100

    colors1 = sns.color_palette('Set1')
    colors2 = sns.color_palette('Dark2')
    colorsSet = colors1[1:5] + colors1[6:8] + colors2[:4] + colors2[5:7] + [colors1[0]]
    assert len(colorsSet) == 13

    marker_dict0 = {
        'Mean': 'o', 'KNN': 's', 'SoftImpute': '^', 'MissForest': 'd', 'Ours (Flex-Net)': 'X',
    }
    marker_dict = dict()
    for kk, vv in marker_dict0.items():
        if kk == 'Ours (Flex-Net)':
            marker_dict[kk] = marker_dict0[kk]
            continue
        for tt in ['RF', 'SVM', 'GCN']:
            marker_dict[f'{kk}+{tt}'] = marker_dict0[kk]

    if name == 'MAR':
        fig = plt.figure(figsize=(3.6, 4))
    else:
        fig = plt.figure(figsize=(3, 4))
    ax = fig.add_subplot(1, 1, 1)

    sns.lineplot(
        ax=ax, data=df, x='lost_num', y='acc', hue='method',
        errorbar=err, err_kws={'linewidth': 0},
        palette=colorsSet,
        style='method', markers=marker_dict, dashes=False,
        linewidth=1.5, sizes=10, markeredgecolor=None,
    )

    ax.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    if leg:
        ax.legend(
            loc="center left", fontsize=10, borderaxespad=0.2, bbox_to_anchor=(1.01, 0.5),
            ncol=1, handlelength=2, columnspacing=0.5, handletextpad=0.5, borderpad=0.5
        )
    else:
        ax.legend_.remove()

    ax.set_facecolor('None')
    ax.grid(axis='both', alpha=0.5, lw=0.5, zorder=1)
    if name == 'MCAR':
        ax.set_ylim(0.715, 0.85)
        ax.set_xlim(4, 16)
    elif name == 'MAR':
        ax.set_ylim(0.645, 0.855)
        ax.set_yticks([0.65, 0.70, 0.75, 0.80, 0.85])
        ax.set_xlim(8, 42)
    elif name == 'MNAR':
        ax.set_ylim(0.715, 0.85)
        ax.set_xlim(4, 16)

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Matrix missing ratio (%)', fontsize=12)
    ax.set_title(name, fontsize=12)

    plt.savefig(f'./figures/simu/{name}.Line.svg', bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()
