# This script plots various variables against each other, Data is retrieved from ResultsOverview.csv file

# Import standard packages
from pathlib import Path                 # Used to manage path variables in windows or linux
import numpy as np                       # Used to do arrays (matrices) computations namely
import pandas as pd                      # Used to manage data frames
import matplotlib.pyplot as plt          # Used to perform plots
import statsmodels.formula.api as smf    # Used for statistical analysis (ols here)
import os
from scipy.stats.distributions import t  # Used to compute confidence intervals
import sys
import seaborn as sns
from scipy.stats import linregress
from matplotlib import cm
from matplotlib.colors import ListedColormap

# Set directory & load data
Cwd = Path.cwd()
DataPath = Cwd / '04_Results/ResultsOverview.csv'
savepath = Cwd / '04_Results/04_Plots/'

df = pd.read_csv(str(DataPath), skiprows=0).dropna().reset_index(drop=True)
df = df.drop(columns=['Sample ID', 'Age / y', 'Gender', 'Ultimate Force / N', 'Organic Weight / g', 'Mineral Weight / g',
                      'Water Weight / g', 'Minimum Equivalent Diameter / mm', 'Mean Apparent Diameter / mm',
                      'Mean Area Fraction / -', 'Min Area Fraction / -', 'Minimum Area / mm²', 'Mean Apparent Area / mm²'])
df_new = df[['Bone Volume Fraction / -', 'Bone Mineral Density / mg HA / cm³', 'Tissue Mineral Density / mg HA / cm³',
             'Mineral to Matrix Ratio / -', 'Mineral weight fraction / -', 'Organic weight fraction / -',
             'Water weight fraction / -', 'Density / g/cm³', 'Apparent Modulus Mineralized / MPa',
             'Apparent Modulus Demineralized / MPa', 'Ultimate Stress / MPa', 'Ultimate Strain / -',
             'Apparent Modulus Mineralized uFE / MPa', 'Yield Stress uFE / MPa', 'Ultimate Stress uFE / MPa']]
df_new.columns = ['Bone Volume Fraction', 'Bone Mineral Density', 'Tissue Mineral Density', 'Mineral to Matrix Ratio',
                  'Mineral Weight Fraction', 'Organic Weight Fraction', 'Water Weight Fraction', 'Bone Density',
                  'Apparent Modulus Mineralized', 'Apparent Modulus Demineralized', 'Ultimate Stress', 'Ultimate Strain',
                  'Apparent Modulus Mineralized uFE', 'Yield Stress uFE', 'Ultimate Stress uFE']
corr_matrix_p = df_new.corr()
corr_matrix_r = df_new.corr()

for i in df_new.columns:
    for j in df_new.columns:
        pvalue = round(linregress(df_new[j], df_new[i])[3], 3)
        if pvalue <= 0.001:
            p = r'\textsuperscript{***}'
        elif pvalue <= 0.01:
            p = r'\textsuperscript{**}'
        elif pvalue <= 0.05:
            p = r'\textsuperscript{*}'

        # corr_matrix.loc[i, j] = round(linregress(df_new[j], df_new[i])[2], 3)
        corr_matrix_p.loc[i, j] = round(linregress(df_new[j], df_new[i])[3], 3)
        corr_matrix_r.loc[i, j] = round(linregress(df_new[j], df_new[i])[2], 3)

# test = linregress(df_new['Ultimate Stress'], df_new['Ultimate Strain'])

p_matrix = pd.DataFrame()
mask_p = np.zeros_like(corr_matrix_p, dtype=np.bool_)
mask_p[np.triu_indices_from(mask_p)] = True

# Colormap trick
viridis_p = cm.get_cmap('plasma', 3)
newcolors_p = viridis_p(np.linspace(0, 1, 3))
newcmp_p = ListedColormap(newcolors_p)

viridis_r = cm.get_cmap('plasma', 8)
newcolors_r = viridis_r(np.linspace(0, 1, 8))
newcmp_r = ListedColormap(newcolors_r)

# abbreviations = ['BVTV', 'BMD', 'TMD', 'MMR', 'WF\u2098', 'WF\u2092', 'WFw', 'db', 'E\u2098', 'Ec', '\u03C3u', '\u03B5u']
abbreviations = ['BVTV', 'BMD', 'TMD', 'MMR', 'WF$_m$', 'WF$_o$', 'WF$_w$', 'd$_b$', 'E$_m$', 'E$_c$', '$\sigma_u$',
                 '$\epsilon_u$', 'E$_{m, \mu FE}$', '$\sigma_{y, \mu FE}$', '$\sigma_{u, \mu FE}$']

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "10"

f, ax = plt.subplots(figsize=(11, 15))
heatmap_p = sns.heatmap(corr_matrix_p,
                        mask=mask_p,
                        square=True,
                        linewidths=.5,
                        cmap=newcmp_p,
                        cbar_kws={'shrink': .5,
                                  'ticks': np.array([1/6, 3/6, 5/6])},
                        vmin=0,
                        vmax=1,
                        annot=True,
                        annot_kws={'size': 12})

ax.collections[0].colorbar.set_ticklabels([0.001, 0.01, 0.05])


#add the column names as labels
# ax.set_yticklabels(corr_matrix.columns, rotation=0, fontsize=14)
# ax.set_xticklabels(corr_matrix.columns, fontsize=14)
ax.set_yticklabels(abbreviations, rotation=0, fontsize=14)
ax.set_xticklabels(abbreviations, fontsize=14)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
plt.savefig(os.path.join(savepath, 'correlation_matrix_heatmap_pvalues.eps'), dpi=300, bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath, 'correlation_matrix_heatmap_pvalues.png'), dpi=600, bbox_inches='tight', format='png')

plt.show()

f, ax = plt.subplots(figsize=(11, 15))
heatmap_r = sns.heatmap(corr_matrix_r,
                        mask=mask_p,
                        square=True,
                        linewidths=.5,
                        cmap=newcmp_r,
                        cbar_kws={'shrink': .5,
                                  'ticks': np.linspace(0, 1, 9)},
                        vmin=0,
                        vmax=1,
                        annot=True,
                        annot_kws={'size': 12})

#add the column names as labels
# ax.set_yticklabels(corr_matrix.columns, rotation=0, fontsize=14)
# ax.set_xticklabels(corr_matrix.columns, fontsize=14)
ax.set_yticklabels(abbreviations, rotation=0, fontsize=14)
ax.set_xticklabels(abbreviations, fontsize=14)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
plt.savefig(os.path.join(savepath, 'correlation_matrix_heatmap_rvalues.eps'), dpi=300, bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath, 'correlation_matrix_heatmap_rvalues.png'), dpi=600, bbox_inches='tight', format='png')

# plt.savefig(os.path.join(savepath, 'correlation_matrix_heatmap.png'), dpi=300, bbox_inches='tight')

plt.show()
