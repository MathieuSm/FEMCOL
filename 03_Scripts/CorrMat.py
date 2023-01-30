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

# Set directory & load data. Remove unwanted columns and rename the remaining ones
Cwd = Path.cwd()
DataPath = Cwd / '04_Results/ResultsOverview.csv'
savepath = Cwd / '04_Results/04_Plots/'

df = pd.read_csv(str(DataPath), skiprows=0)
df = df.drop(columns=['Sample ID', 'Age / y', 'Gender', 'Ultimate Force / N', 'Organic Weight / g', 'Mineral Weight / g',
                      'Water Weight / g', 'Min Equivalent Diameter / mm', 'Mean Apparent Diameter / mm',
                      'Mean Area Fraction / -', 'Min Area Fraction / -', 'Mean Apparent Area / mm²'])
df_new = df[['Bone Volume Fraction / -', 'Bone Mineral Density / mg HA / cm³', 'Tissue Mineral Density / mg HA / cm³',
             'Mineral to Matrix Ratio / -', 'Mineral weight fraction / -', 'Organic weight fraction / -',
             'Water weight fraction / -', 'Density / g/cm³', 'Apparent Modulus Mineralized / MPa',
             'Modulus Mineralized / MPa', 'Apparent Modulus Demineralized / MPa', 'Modulus Demineralized / MPa',
             'Ultimate Apparent Stress / MPa', 'Ultimate Collagen Stress / MPa', 'Ultimate Stress / MPa',
             'Ultimate Strain / -', 'Apparent Modulus Mineralized uFE / MPa', 'Yield Stress uFE / MPa',
             'Ultimate Stress uFE / MPa', 'Min ECM Area / mm²', 'Coefficient of Variance / -']]
# df_new.columns = ['Bone Volume Fraction', 'Bone Mineral Density', 'Tissue Mineral Density', 'Mineral to Matrix Ratio',
#                   'Mineral Weight Fraction', 'Organic Weight Fraction', 'Water Weight Fraction', 'Bone Density',
#                   'Apparent Modulus Mineralized', 'Apparent Modulus Demineralized', 'Ultimate Stress', 'Ultimate Strain',
#                   'Apparent Modulus Mineralized uFE', 'Yield Stress uFE', 'Ultimate Stress uFE']
df_new.columns = ['BVTV', 'BMD', 'TMD', 'MMR', 'WFM', 'WFO', 'WFW', 'D', 'AMM', 'MM', 'AMD', 'MD', 'UAPPSTRE', 'UCSTRE',
                  'USTRE', 'USTRA', 'AMMuFE', 'YSTREuFE', 'USTREuFE', 'MINECMA', 'COFVAR']

# Create empty dataframe for p-values and correlation matrix containing r values
corr_matrix_p = pd.DataFrame()
corr_matrix_r = round(df_new.corr(), 2)
test_matrix_r = pd.DataFrame()

# loop to iterate through rows and columns
for i in df_new.columns:
    for j in df_new.columns:

        # mask to select only values that are not NaN (linregress does not work otherwise)
        mask = ~df_new[i].isna() & ~df_new[j].isna()
        slope, intercept, r, pvalue, std_err = linregress(df_new[j][mask], df_new[i][mask])

        # create matrix containing p-values using mask criterion
        # corr_matrix_p.loc[i, j] = round(linregress(df_new[j][mask], df_new[i][mask])[3], 3)
        corr_matrix_p.loc[i, j] = pvalue

        # pvalue = round(linregress(df_new[j][mask], df_new[i][mask])[3], 10)
        if pvalue <= 0.001:
            p = r'\textsuperscript{***}'
            test_matrix_r.loc[i, j] = str(corr_matrix_r.loc[i, j]) + str(p)
        elif pvalue <= 0.01:
            p = r'\textsuperscript{**}'
            test_matrix_r.loc[i, j] = str(corr_matrix_r.loc[i, j]) + str(p)
        elif pvalue <= 0.05:
            p = r'\textsuperscript{*}'
            test_matrix_r.loc[i, j] = str(corr_matrix_r.loc[i, j]) + str(p)
        elif pvalue > 0.05:
            test_matrix_r.loc[i, j] = str(corr_matrix_r.loc[i, j])

# Masking upper triangle to show only lower triangle for p-plot without asterisk
mask_p = np.zeros_like(corr_matrix_p, dtype=np.bool_)
mask_p[np.triu_indices_from(mask_p)] = True

# Masking upper triangle to show only lower triangle for p-plot with asterisk
mask_r_red = np.zeros_like(corr_matrix_r, dtype=np.bool_)
mask_r_red[np.tril_indices_from(mask_r_red)] = True
corr_matrix_r_red = (test_matrix_r.mask(mask_r_red)).fillna('')


# Colormap trick for p-matrix
viridis_p = cm.get_cmap('plasma', 4)
newcolors_p = viridis_p(np.linspace(0, 1, 4))
newcmp_p = ListedColormap(newcolors_p)

# # Colormap trick for r-matrix
# viridis_r = cm.get_cmap('plasma', 8)
# newcolors_r = viridis_r(np.linspace(0, 1, 8))
# newcmp_r = ListedColormap(newcolors_r)

twilight = cm.get_cmap('twilight', 8)
newcolors_r = twilight(np.linspace(0, 1, 8))
newcmp_r = ListedColormap(newcolors_r)

# Axis annotations
abbreviations = ['BVTV', 'BMD', 'TMD', 'MMR', 'WF$_m$', 'WF$_o$', 'WF$_w$', 'd$_b$', 'E$_{app, m}$', 'E$_m$',
                 'E$_{app, c}$', 'E$_c$', '$\sigma_{app}$', '$\sigma_c$', '$\sigma_b$','$\epsilon_c$',
                 'E$_{m, \mu FE}$', '$\sigma_{y, \mu FE}$', '$\sigma_{u, \mu FE}$', '$AF_{min}$', '$CV_{AF_{min}}$']

# Font style and size
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "10"

## Added to "trick" the plot
Trick = corr_matrix_p.copy()
Steps = [0, 0.001, 0.01, 0.05, 1]
for i, Step in enumerate(Steps):
    Trick[Trick < Step] = i
Trick = Trick / len(Steps)

f, ax = plt.subplots(figsize=(13, 17))
heatmap_p = sns.heatmap(Trick,
                        mask=mask_p,
                        square=True,
                        linewidths=.5,
                        cmap=newcmp_p,
                        cbar_kws={'shrink': .5,
                                  'ticks': np.array([0, 1/4, 2/4, 3/4, 1]),
                                  'label': 'p-value'},
                        vmin=0,
                        vmax=1,
                        fmt='.3f',
                        annot=corr_matrix_p,
                        annot_kws={'size': 10})

# set ticklabels of colorbar
ax.collections[0].colorbar.set_ticklabels([0, 0.001, 0.01, 0.05, 1])

#add column names as labels
ax.set_yticklabels(abbreviations, rotation=0, fontsize=14)
ax.set_xticklabels(abbreviations, fontsize=14)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

plt.savefig(os.path.join(savepath, 'correlation_matrix_heatmap_pvalues.eps'), dpi=600, bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath, 'correlation_matrix_heatmap_pvalues.png'), dpi=600, bbox_inches='tight', format='png')

plt.show()

# Plotting of r-values
mask_r = np.zeros_like(corr_matrix_r, dtype=np.bool_)
mask_r[np.triu_indices_from(mask_r)] = True

f, ax = plt.subplots(figsize=(13, 17))
heatmap_r = sns.heatmap(corr_matrix_r,
                        mask=mask_r,
                        square=True,
                        linewidths=.5,
                        cmap=newcmp_r,
                        cbar_kws={'shrink': .5,
                                  'ticks': np.linspace(-1, 1, 9),
                                  'label': 'Pearson Correlation Coefficient (r)'},
                        vmin=-1,
                        vmax=1,
                        annot=True,
                        fmt='.2f',
                        annot_kws={'size': 10})

#add the column names as labels
ax.set_yticklabels(abbreviations, rotation=0, fontsize=14)
ax.set_xticklabels(abbreviations, fontsize=14)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
plt.savefig(os.path.join(savepath, 'correlation_matrix_heatmap_rvalues.eps'), dpi=600, bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath, 'correlation_matrix_heatmap_rvalues.png'), dpi=600, bbox_inches='tight', format='png')

plt.show()

# Plotting of r-values containing p-value asterisk
f, ax = plt.subplots(figsize=(13, 17))
heatmap_r = sns.heatmap(corr_matrix_r,
                        mask=mask_r,
                        square=True,
                        linewidths=.5,
                        cmap=newcmp_r,
                        cbar_kws={'shrink': .5,
                                  'ticks': np.linspace(-1, 1, 9),
                                  'label': 'Pearson Correlation Coefficient (r)'},
                        vmin=-1,
                        vmax=1,
                        fmt='.2f',
                        annot=False,
                        annot_kws={'size': 10})

for i, c in enumerate(corr_matrix_r_red.columns):
    for j, v in enumerate(corr_matrix_r_red[c]):
        ax.text(i + 0.5, j + 0.5, str(corr_matrix_r_red.iloc[i, j]), ha='center', va='center', color='w')

#add column names as labels
ax.set_yticklabels(abbreviations, rotation=0, fontsize=14)
ax.set_xticklabels(abbreviations, fontsize=14)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

plt.savefig(os.path.join(savepath, 'correlation_matrix_heatmap_rvalues_asterisk.eps'), dpi=600, bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath, 'correlation_matrix_heatmap_rvalues_asterisk.png'), dpi=600, bbox_inches='tight', format='png')

plt.show()
