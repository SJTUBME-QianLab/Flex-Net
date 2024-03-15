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
    # Fig. 4 a
    plot_characteristics()

    save_dict = cal_ave()
    ave = save_dict['Ave-acc-addSOTA']
    std = save_dict['Std-acc-addSOTA']
    rank = cal_rank(ave.copy())
    save_dict['Ave-acc-rank30'] = rank

    pd_writer = pd.ExcelWriter('./UCI_AveStd.xlsx')
    for kk, vv in save_dict.items():
        vv.to_excel(pd_writer, sheet_name=kk, index=True)
    pd_writer.save()

    # Fig. 4 b
    ave = RenameDF(ave.copy())
    std = RenameDF(std.copy())
    plot_DotLine(ave, std)
    
    # Fig. 4 c
    rank = RenameDF(rank.copy())
    print(rank.shape)
    plot_BubbleRank(rank)


def plot_characteristics():
    df = pd.read_excel('./all_results.xlsx', sheet_name='UCI_datasets')

    fig = plt.figure(figsize=(16, 4))

    cmap_256 = plt.get_cmap('gist_earth', 10)
    colors2 = [cmap_256(kk) for kk in range(1, 9)]
    width = 0.7

    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        wspace=0.15, hspace=0.4
    )
    x = df['Abbrev']
    axes_dict = dict()

    # Sample
    ax1 = fig.add_subplot(gs[0, 0])
    axes_dict['Sample'] = ax1
    ya = [int(kk.split(' ')[0]) for kk in df['Sample (More/Less)']]
    y0 = [int(kk.split(' (')[1].split('/')[0]) for kk in df['Sample (More/Less)']]
    y1 = [int(kk.split('/')[1].split(')')[0]) for kk in df['Sample (More/Less)']]
    sns.barplot(
        ax=ax1, x=df['Abbrev'], palette=colors2,
        y=ya, label='Less', alpha=0.6,  # edgecolor='gray'
        width=width,
    )  # total
    sns.barplot(
        ax=ax1, x=df['Abbrev'], palette=colors2, zorder=2,
        y=y0, label='More',  # edgecolor='gray'
        width=width,
    )  # more
    for i in range(len(df)):
        ax1.text(
            x=i, y=ya[i], s=int(ya[i]),
            horizontalalignment='center', verticalalignment='bottom',
        )

    # Characteristic
    ax2 = fig.add_subplot(gs[0, 1])
    axes_dict['Attribute'] = ax2
    y = df['Attribute']
    sns.barplot(ax=ax2, x=x, y=y, palette=colors2, width=width, zorder=2)
    for i in range(len(df)):
        ax2.text(
            x=i, y=y[i], s=int(y[i]),
            horizontalalignment='center', verticalalignment='bottom',
        )

    # Missing patterns
    ax3 = fig.add_subplot(gs[0, 2])
    axes_dict['Missing pattern'] = ax3
    y = df['Missing pattern']
    sns.barplot(ax=ax3, x=x, y=y, palette=colors2, width=width, zorder=2)
    for i in range(len(df)):
        ax3.text(
            x=i, y=df['Missing pattern'][i], s=int(df['Missing pattern'][i]),
            horizontalalignment='center', verticalalignment='bottom',
        )

    # Incomplete sample ratio
    ax4 = fig.add_subplot(gs[1, 0])
    axes_dict['Ratio of incomplete sample ($R_s$)'] = ax4
    y = df['Incomplete sample ratio']
    sns.barplot(ax=ax4, x=x, y=y, palette=colors2, width=width, zorder=2)
    for i in range(len(df)):
        ax4.text(
            x=i, y=y[i], s='%.3f' % y[i],
            horizontalalignment='center', verticalalignment='bottom',
        )

    # Incomplete attribute ratio
    ax5 = fig.add_subplot(gs[1, 1])
    axes_dict['Ratio of incomplete attribute ($R_a$)'] = ax5
    y = df['Incomplete attribute ratio']
    sns.barplot(ax=ax5, x=x, y=y, palette=colors2, width=width, zorder=2)
    for i in range(len(df)):
        ax5.text(
            x=i, y=y[i], s='%.3f' % y[i],
            horizontalalignment='center', verticalalignment='bottom',
        )

    # Matrix missing ratio
    ax6 = fig.add_subplot(gs[1, 2])
    axes_dict['Matrix missing ratio ($R_m$)'] = ax6
    y = df['Matrix missing ratio']
    sns.barplot(ax=ax6, x=x, y=y, palette=colors2, width=width, zorder=2)
    for i in range(len(df)):
        ax6.text(
            x=i, y=y[i], s='%.3f' % y[i],
            horizontalalignment='center', verticalalignment='bottom',
        )

    ax1.set_ylim(0, 800 * 1.1)
    ax1.set_yticks(ticks=np.linspace(start=0, stop=800, num=3, endpoint=True))
    ax2.set_ylim(0, 50 * 1.1)
    ax2.set_yticks(ticks=np.linspace(start=0, stop=50, num=3, endpoint=True))
    ax3.set_ylim(0, 240 * 1.1)
    ax3.set_yticks(ticks=np.linspace(start=0, stop=240, num=3, endpoint=True))
    ax4.set_ylim(0, 1.15)
    ax4.set_yticks(ticks=np.linspace(start=0, stop=1, num=3, endpoint=True))
    ax5.set_ylim(0, 1.15)
    ax5.set_yticks(ticks=np.linspace(start=0, stop=1, num=3, endpoint=True))
    ax6.set_ylim(0, 0.27)
    ax6.set_yticks(ticks=np.linspace(start=0, stop=0.2, num=3, endpoint=True))

    for name, ax in axes_dict.items():
        ax.tick_params(
            axis='both', labelsize=12, length=2, width=1.5,
            bottom=False, top=False, left=True, right=False,
            labelbottom=True, labeltop=False, labelleft=True, labelright=False,
        )
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_visible(False)

        ax.set_facecolor('None')  # 透明背景
        ax.grid(axis='y', alpha=0.5, lw=0.5, zorder=1)

        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(name, fontsize=12)

    plt.savefig('./figures/Fig4_Info.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def TTestPV(aa, bb, alternative='two-sided'):
    assert alternative in ['two-sided', 'greater', 'less']
    return scipy.stats.ttest_rel(aa, bb, alternative=alternative).pvalue


def cal_ave_UCI(data_name, methods_list):
    df = pd.read_excel('./all_results.xlsx', sheet_name=data_name)
    # assert df.shape == ((4*3+1)*10, 2)
    # print(df.shape)
    assert set(df['method'].to_list()) == set(methods_list)

    acc = pd.DataFrame(columns=['Ave', 'Std', 'Ttest'], index=methods_list)
    for mm in methods_list:
        acc.loc[mm, 'Ave'] = np.mean(df[df['method'] == mm]['acc'].astype(float), axis=0)
        acc.loc[mm, 'Std'] = np.std(df[df['method'] == mm]['acc'].astype(float), axis=0)

    Ours = df[df['method'] == 'Ours']['acc'].astype(float)
    for mm in methods_list:
        target = df[df['method'] == mm]['acc'].astype(float)
        acc.loc[mm, 'Ttest'] = TTestPV(Ours, target, 'greater')
        # two sample paired one-tailed T-test
    return acc


def divideAveStd(data):
    methods = list(data.columns)
    ave = pd.DataFrame(columns=data.index, index=methods)
    std = pd.DataFrame(columns=data.index, index=methods)
    for name in data.index:
        for mm in methods:
            value = data.loc[name, mm]
            if isinstance(value, int):
                value = float(value)
            if isinstance(value, float):
                if np.isnan(value):
                    ave.loc[mm, name] = np.nan
                    std.loc[mm, name] = np.nan
                else:
                    ave.loc[mm, name] = value
                    std.loc[mm, name] = 0
            elif isinstance(value, str):
                assert '±' in value
                ave.loc[mm, name] = float(value.split('±')[0])
                std.loc[mm, name] = float(value.split('±')[1])
    return ave, std


def cal_ave():
    fill_methods = ['mean', 'softimpute', 'knn.3', 'missForest']
    methods_list = [f'{kk}+{tt}' for kk in fill_methods for tt in ['RF', 'SVM', 'GCN']] + ['Ours']

    data_list = ['BC', 'CC', 'CK', 'HC', 'HD', 'HP', 'HS', 'PI']
    save_dict = dict()
    for data_name in data_list:
        # print(data_name)
        acc = cal_ave_UCI(data_name, methods_list)
        save_dict[data_name] = acc
    for met in ['Ave', 'Std', 'Ttest']:
        save_dict[f'{met}-acc'] = pd.DataFrame(dict(zip(
            data_list,
            [save_dict[dd][met].values for dd in data_list],
        )), index=methods_list)

    sota = pd.read_excel('./all_results.xlsx', sheet_name='UCI_SOTA', index_col=0)
    ave_sota, std_sota = divideAveStd(sota)

    ave = save_dict['Ave-acc'].copy()
    for i in range(ave.shape[0]):
        for j in range(ave.shape[1]):
            ave.iloc[i, j] = np.round(ave.iloc[i, j] * 100, 2)
    ours = ave.loc[['Ours'], :]
    ave = ave.drop(['Ours'], axis=0)
    ave = pd.concat([ours, ave, ave_sota], axis=0)

    std = save_dict['Std-acc'].copy()
    for i in range(std.shape[0]):
        for j in range(std.shape[1]):
            std.iloc[i, j] = np.round(std.iloc[i, j] * 100, 2)
    ours = std.loc[['Ours'], :]
    std = std.drop(['Ours'], axis=0)
    std = pd.concat([ours, std, std_sota], axis=0)

    save_dict['Ave-acc-addSOTA'] = ave
    save_dict['Std-acc-addSOTA'] = std
    return save_dict


def RenameDF(df):
    fill_methods = {
        'mean': 'Mean', 'knn.3': 'KNN', 'softimpute': 'SoftImpute', 'missForest': 'MissForest'
    }
    dfR = pd.DataFrame(columns=df.columns)
    dfR.loc['Ours (Flex-Net)'] = df.loc['Ours', :]

    for tt in ['RF', 'SVM', 'GCN']:
        for i, kk in enumerate(list(fill_methods.keys())):
            dfR.loc[f'{fill_methods[kk]}+{tt}'] = df.loc[f'{kk}+{tt}', :]

    dfR.loc['', :] = np.nan
    dfR = pd.concat([dfR, df.iloc[13:, :]], axis=0)

    dfR.index = [kk.replace('Tran (', 'Tran (+') for kk in dfR.index]

    return dfR


def get_colors(nan=True):
    colors1 = sns.color_palette('Set3')
    colors2 = sns.color_palette('tab10')
    colorsSet = [colors1[0]] + [colors1[2]] + colors1[4:12] + colors2[-2:] + [colors1[3]]
    colors = []
    for i in range(3):
        colors.extend(colorsSet[:4])
    if nan:
        colors.extend([colorsSet[-1]])
    colors.extend([colorsSet[10]] * 4)
    colors.extend([colorsSet[6]] * 5)
    colors.extend([colorsSet[9]] * 5)
    colors.extend(colorsSet[7:9])
    colors.extend(colorsSet[-2:])
    colors = [colors[-1]] + colors[:-1]
    return colors


def plot_DotLine(ave, std):
    colors = get_colors()
    ave = ave / 100
    std = std / 100
    assert len(colors) == len(ave)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(
        nrows=1, ncols=ave.shape[1],
        wspace=0.11,
    )
    axes = []
    for i, name in enumerate(ave.columns):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)

        ax.hlines(
            np.arange(len(ave)),
            xmin=ave[name] - std[name],
            xmax=ave[name] + std[name],
            edgecolor='0.', linewidths=3,
            zorder=2
        )
        ax.scatter(
            y=ave.index, x=ave[name],
            c=colors, s=50, marker='d',
            edgecolor='0.', linewidths=1.5,
            zorder=3
        )
        ax.set_ylim(len(ave) - 0.2, -0.8)
        xmin = np.nanmin((ave[name] - std[name]).values)
        xmax = np.nanmax((ave[name] + std[name]).values)
        print(name, xmin, xmax)

        ax.grid(axis='y', alpha=0.5, lw=0.5, zorder=1)
        # Ours
        ax.vlines(
            ave.loc['Ours (Flex-Net)', name],
            ymin=-0.8, ymax=len(ave) - 0.2,
            edgecolor='firebrick', linewidths=0.5, linestyle='--',
            zorder=1
        )

        ax.tick_params(
            axis='both', labelsize=12, length=2, width=1.5,
            bottom=True, top=False, left=False, right=False,
            labelbottom=True, labeltop=False, labelleft=False, labelright=False,
        )

        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_visible(False)
        ax.set_facecolor('None')

        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(name, fontsize=12)

    axes[0].tick_params(labelleft=True, left=False)
    for i, label in enumerate(axes[0].get_yticklabels()):
        if i == 0:
            print(label)
            label.set_color('firebrick')

    axes[4].text(
        0.72, 32, 'Accuracy',
        fontsize=12, horizontalalignment='center', verticalalignment='top',
    )
    axes[0].set_xlim(0.75, 1.00)
    axes[0].set_xticks([0.80, 0.88, 0.96], ['0.80', '0.88', '0.96'])
    axes[0].hlines(y=13, xmin=0.75, xmax=1.00, color='grey', linewidths=2, linestyle='--')
    axes[1].set_xlim(0.94, 1.00)
    axes[1].set_xticks([0.95, 0.97, 0.99], ['0.95', '0.97', '0.99'])
    axes[1].hlines(y=13, xmin=0.94, xmax=1.00, color='grey', linewidths=2, linestyle='--')
    axes[2].set_xlim(0.74, 1.02)
    axes[2].set_xticks([0.80, 0.90, 1.00], ['0.80', '0.90', '1.00'])
    axes[2].hlines(y=13, xmin=0.74, xmax=1.02, color='grey', linewidths=2, linestyle='--')
    axes[3].set_xlim(0.58, 0.89)
    axes[3].set_xticks([0.65, 0.75, 0.85], ['0.65', '0.75', '0.85'])
    axes[3].hlines(y=13, xmin=0.58, xmax=0.89, color='grey', linewidths=2, linestyle='--')
    axes[4].set_xlim(0.725, 0.86)
    axes[4].set_xticks([0.75, 0.80, 0.85], ['0.75', '0.80', '0.85'])
    axes[4].hlines(y=13, xmin=0.725, xmax=0.86, color='grey', linewidths=2, linestyle='--')
    axes[5].set_xlim(0.60, 0.94)
    axes[5].set_xticks([0.70, 0.80, 0.90], ['0.70', '0.80', '0.90'])
    axes[5].hlines(y=13, xmin=0.60, xmax=0.94, color='grey', linewidths=2, linestyle='--')
    axes[6].set_xlim(0.44, 0.77)
    axes[6].set_xticks([0.50, 0.60, 0.70], ['0.50', '0.60', '0.70'])
    axes[6].hlines(y=13, xmin=0.44, xmax=0.77, color='grey', linewidths=2, linestyle='--')
    axes[7].set_xlim(0.69, 0.82)
    axes[7].set_xticks([0.70, 0.75, 0.80], ['0.70', '0.75', '0.80'])
    axes[7].hlines(y=13, xmin=0.69, xmax=0.82, color='grey', linewidths=2, linestyle='--')

    plt.savefig('./figures/Fig4_DotLine.svg', bbox_inches='tight', pad_inches=0)
    plt.show()


def cal_rank(ave):
    rank = ave.copy()
    for kk in ave.columns:
        rank[kk] = ave[kk].rank(ascending=False)
    rank['mean'] = rank.mean(axis=1)
    rank['std'] = rank.std(axis=1)
    return rank


def plot_BubbleRank(df):
    df['method'] = df.index
    df.reset_index(drop=True, inplace=True)
    colors = get_colors(nan=False)
    print(len(colors), len(df))

    fig = plt.figure(figsize=(14, 2))
    ax = fig.add_subplot(1, 1, 1)

    amp, zero = 1.5, 1.2
    minsize = np.pi * (df['std'].min() * amp + zero) ** 2
    maxsize = np.pi * (df['std'].max() * amp + zero) ** 2
    print(len(df['mean']))
    sns.scatterplot(
        data=df, y='mean', x='method', size='std',
        sizes=(minsize, maxsize),
        color=colors, alpha=1, edgecolors='black',
        zorder=2,
    )

    ax.set_xlim(-0.8, len(df) - 0.2)
    ax.set_xticks(
        np.arange(len(df)), df['method'],
        rotation=40, ha='right', rotation_mode='anchor'
    )
    ax.set_ylim(0, 22)
    ax.set_yticks([0, 5, 10, 15, 20])
    for i in range(len(df)):
        if i == 13:
            ax.vlines(x=13, ymin=0, ymax=22, color='grey', linewidths=2, linestyle='--')
        else:
            ax.vlines(x=i, ymin=0, ymax=df['mean'][i], color='grey', linewidths=1, linestyle='-', zorder=1)

    ax.set_xlabel('')
    ax.set_ylabel('Rank', fontsize=12)
    ax.legend(
        title='Std',
        loc='upper left', fontsize=10, bbox_to_anchor=(1, 1),
        ncol=1, columnspacing=0.5, borderpad=0.9, labelspacing=1.5, handlelength=1.3, handletextpad=1
    )

    ax.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)
    ax.set_facecolor('None')  # 透明背景

    for label in ax.get_xticklabels():
        if label.get_text() == 'Ours (Flex-Net)':
            label.set_color('firebrick')

    plt.savefig('./figures/Fig4_BubbleRank.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


if __name__ == '__main__':
    main()
