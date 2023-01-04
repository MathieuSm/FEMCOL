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

# Set directory & load data
Cwd = Path.cwd()
DataPath = Cwd / '04_Results/ResultsOverview.csv'
savepath = Cwd / '04_Results/04_Plots/CorrelationMatrix'
# DataPath = 'C:/Users/Stefan/Dropbox/02_MScThesis/09_Results/ResultsOverview.csv'
df = pd.read_csv(str(DataPath), skiprows=0).dropna().reset_index(drop=True)
df = df.drop(columns=['Sample ID', 'Age', 'Gender', 'Ultimate Force N', 'Organic Weight g', 'Mineral Weight g',
                      'Water Weight g', 'Minimum Equivalent Diameter mm', 'Mean Apparent Diameter mm',
                      'Mean Area Fraction -', 'Min Area Fraction -', 'Minimum Area mm²', 'Mean Apparent Area mm²'])
df_new = df[['Bone Volume Fraction -', 'Bone Mineral Density mg HA / cm³', 'Tissue Mineral Density mg HA / cm³',
             'Mineral to Matrix Ratio -', 'Mineral weight fraction -', 'Organic weight fraction -',
             'Water weight fraction -', 'Density g/cm³', 'Apparent Modulus Mineralized MPa',
             'Apparent Modulus Demineralized MPa', 'Ultimate Stress MPa', 'Ultimate Strain -']]
df_new.columns = ['Bone Volume Fraction', 'Bone Mineral Density', 'Tissue Mineral Density', 'Mineral to Matrix Ratio',
                  'Mineral Weight Fraction', 'Organic Weight Fraction', 'Water Weight Fraction', 'Bone Density',
                  'Apparent Modulus Mineralized', 'Apparent Modulus Demineralized', 'Ultimate Stress', 'Ultimate Strain']
corr_matrix = df_new.corr()

for i in df_new.columns:
    for j in df_new.columns:
        pvalue = round(linregress(df_new[j], df_new[i])[3], 3)
        if pvalue <= 0.001:
            p = str('***')
        elif pvalue <= 0.01:
            p = str('**')
        elif pvalue <= 0.05:
            p = str('*')
        #corr_matrix.loc[i, j] = round(linregress(df_new[j], df_new[i])[2], 3) + p
        corr_matrix.loc[i, j] = round(linregress(df_new[j], df_new[i])[2], 3)

p_matrix = pd.DataFrame()
mask = np.zeros_like(corr_matrix, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True

# Colormap trick
from matplotlib import cm
from matplotlib.colors import ListedColormap
viridis = cm.get_cmap('plasma', 8)
newcolors = viridis(np.linspace(0, 1, 8))
newcmp = ListedColormap(newcolors)

# abbreviations = ['BVTV', 'BMD', 'TMD', 'MMR', 'WF\u2098', 'WF\u2092', 'WFw', 'db', 'E\u2098', 'Ec', '\u03C3u', '\u03B5u']
abbreviations = ['BVTV', 'BMD', 'TMD', 'MMR', 'WF$_m$', 'WF$_o$', 'WF$_w$', 'd$_b$', 'E$_m$', 'E$_c$', '$\sigma_u$',
                 '$\epsilon_u$']

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "10"

f, ax = plt.subplots(figsize=(11, 15))
heatmap = sns.heatmap(corr_matrix,
                      mask=mask,
                      square=True,
                      linewidths=.5,
                      cmap=newcmp,
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
plt.savefig(os.path.join(savepath, 'correlation_matrix_heatmap.eps'), dpi=300, bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath, 'correlation_matrix_heatmap.png'), dpi=300, bbox_inches='tight', format='png')
# plt.savefig(os.path.join(savepath, 'correlation_matrix_heatmap.png'), dpi=300, bbox_inches='tight')

plt.show()

