# This script plots various variables against each other in a correlation matrix, Data is retrieved from
# ResultsOverview.csv file

# Import standard packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import linregress
from matplotlib import cm
from matplotlib.colors import ListedColormap


# Set directory & load data. Remove unwanted columns and rename the remaining ones
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
results_path = str(__location__ + '/04_Results')
results_overview = str(results_path + '/ResultsOverview.csv')
savepath_full = results_path + '/04_Plots/CorrelationMatrix_Full/'
savepath_compact = results_path + '/04_Plots/CorrelationMatrix_Compact/'

df = pd.read_csv(str(results_overview), skiprows=0)
df = df.drop(columns=['Sample ID', 'Gender', 'Site', 'Stiffness Mineralized / N/mm',
                      'Stiffness Demineralized / N/mm', 'Ultimate Force / N', 'Organic Weight / g', 'Mineral Weight / g',
                      'Water Weight / g', 'Bone Mineral Content / mg HA'])
# # Complete Matrix including all variables
# df_new = df[['Bone Volume Fraction / -', 'Bone Mineral Density / mg HA / cm³', 'Tissue Mineral Density / mg HA / cm³',
#              'Mineral to Matrix Ratio v2/a3 / -', 'Mineral to Matrix Ratio v1/a1 / -', 'Crystallinity / -',
#              'Collagen dis/order / -', 'Matrix maturity / -', 'Mineral weight fraction / -',
#              'Organic weight fraction / -', 'Water weight fraction / -', 'Density / g/cm³',
#              'Apparent Modulus Mineralized / MPa', 'Modulus Mineralized / MPa', 'Apparent Modulus Demineralized / MPa',
#              'Modulus Demineralized / MPa', 'Ultimate Apparent Stress / MPa', 'Ultimate Collagen Stress / MPa',
#              'Ultimate Stress / MPa', 'Ultimate Strain / -', 'Apparent Modulus Mineralized uFE / MPa',
#              'Yield Stress uFE / MPa', 'Ultimate Stress uFE / MPa']]
# df_new = df[['Age / y', 'Mineral weight fraction / -', 'Organic weight fraction / -', 'Water weight fraction / -',
#              'Bone Volume Fraction / -', 'Tissue Mineral Density / mg HA / cm³',
#              'Mineral to Matrix Ratio v2/a3 / -', 'Crystallinity / -', 'Collagen dis/order / -', 'Matrix maturity / -',
#              'Relative Pyridinoline Content / -', 'Relative Proteoglycan Content / -', 'Relative Lipid Content / -',
#              'Modulus Mineralized / MPa', 'Apparent Modulus Mineralized / MPa',
#              'Modulus Demineralized / MPa', 'Apparent Modulus Demineralized / MPa', 'Ultimate Apparent Stress / MPa',
#              'Ultimate Strain / -']]

# Create new, compact dataframe that combines the properties of interest
df_new = df[['Age / y', 'Bone Volume Fraction / -', 'Tissue Mineral Density / mg HA / cm³',
             'Mean Bone Area Fraction / -', 'Min Bone Area Fraction / -',
             'Apparent Modulus Mineralized / GPa', 'Modulus Mineralized / GPa', 'Apparent Modulus Demineralized / MPa',
             'Modulus Demineralized / MPa', 'Ultimate Apparent Stress / MPa', 'Ultimate Strain / -',
             'Mineral weight fraction / -', 'Organic weight fraction / -', 'Water weight fraction / -',
             'Mineral to Matrix Ratio v2/a3 / -', 'Crystallinity / -', 'Collagen dis/order / -', 'Matrix maturity / -',
             'Haversian Canals Mean / %', 'Osteocytes Mean / %', 'Cement Lines Mean / %']]

# # Abbreviations of complete Matrix including all variables (has to be in similar order as df_new)
# df_new.columns = ['Age', 'BVTV', 'BMD', 'TMD', 'MMRv2a3', 'MMRv1a1', 'XC', 'CDO', 'MMAT', 'WFM', 'WFO', 'WFW', 'D',
#                   'AMM', 'MM', 'AMD', 'MD', 'UAPPSTRE', 'UCSTRE', 'USTRE', 'USTRA', 'AMMuFE', 'YSTREuFE', 'USTREuFE',
#                   'MEANECMAF', 'MINECMAF', 'COFVAR']
# df_new.columns = ['Age', 'WFM', 'WFO', 'WFW',
#                   'BVTV', 'TMD',
#                   'MMRv2a3', 'XC', 'CDO', 'MMAT',
#                   'RPyC', 'RPrC', 'RLC',
#                   'MM', 'AMM',
#                   'MD', 'AMD', 'UAPPSTRE', 'USTRA']

# Replace column names of compact matrix with abbreviations, order should be the same as in df_new
df_new.columns = ['Age', 'BV/TV', 'TMD', 'BA/TA_mean', 'BA/TA_min',
                  'AMM', 'MM', 'AMD', 'MD', 'UAPPSTRE', 'USTRA',
                  'WFM', 'WFO', 'WFW',
                  'MMRv2a3', 'XC', 'CDO', 'MMAT',
                  'HC', 'OC', 'CL']

# Create empty dataframe for p-values and correlation matrix containing r values
corr_matrix_p = pd.DataFrame()
corr_matrix_r = round(df_new.corr(), 2)
test_matrix_r = pd.DataFrame()

# loop to iterate through rows and columns
for i in df_new.columns:
    for j in df_new.columns:

        # mask to select only values that are not NaN (linregress does not work otherwise) and perform regression
        mask = ~df_new[i].isna() & ~df_new[j].isna()
        slope, intercept, r, pvalue, std_err = linregress(df_new[j][mask], df_new[i][mask])

        # create matrix containing p-values
        corr_matrix_p.loc[i, j] = pvalue

        # define asterisk criterion for p-values
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

twilight = cm.get_cmap('twilight', 8)
newcolors_r = twilight(np.linspace(0, 1, 8))
newcmp_r = ListedColormap(newcolors_r)

# Axis annotations of complete correlation matrix (need to be in the same order as in df_new)
# abbreviations = ['Age', 'BVTV', 'BMD', 'TMD', r'MMR$\nu_{2}a_{3}$', r'MMR$\nu_{1}a_{1}$', 'X$_c$', 'CDO', 'MMAT',
#                  'WF$_m$', 'WF$_o$', 'WF$_w$', r'$\rho_{b}$', 'E$_{app, m}$', 'E$_m$', 'E$_{app, c}$', 'E$_c$',
#                  '$\sigma_{app}$', '$\sigma_c$', '$\sigma_b$', '$\epsilon_c$', 'E$_{m, \mu FE}$', '$\sigma_{y, \mu FE}$',
#                  '$\sigma_{u, \mu FE}$', '$ECM_{AF_{mean}}$', '$ECM_{AF_{min}}$', '$ECM_{A_{CV}}$']

# Axis annotations of compact correlation matrix (need to be in the same order as in df_new)
abbreviations = ['Age', r'$\rho$', 'TMD', 'BA/TA$_{mean}$', 'BA/TA$_{min}$',
                 'E$_{m}^{app}$', 'E$_m$', 'E$_{c}^{app}$', 'E$_c$', '$\sigma_{u}^{app}$', '$\epsilon_c$',
                 'WF$_m$', 'WF$_o$', 'WF$_w$',
                 r'MMR$\nu_{2}a_{3}$', 'X$_c$', 'CDO', 'MMAT',
                 'HC', 'OC', 'CL']

# Font style and size
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "13"  # 10 for full, 13 for compact corr. matrix

# Coloring trick to show limited number of different colors
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

plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_pvalues_compact.eps'), dpi=1200,
            bbox_inches='tight', format='eps')
# plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_pvalues_full.eps'), dpi=1200,
#             bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_pvalues_compact.png'), dpi=1200,
            bbox_inches='tight', format='png')
# plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_pvalues_full.png'), dpi=1200,
#             bbox_inches='tight', format='png')

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
plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_compact.eps'), dpi=1200,
            bbox_inches='tight', format='eps')
# plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_full.eps'), dpi=1200,
#             bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_compact.png'), dpi=1200,
            bbox_inches='tight', format='png')
# plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_full.png'), dpi=1200,
#             bbox_inches='tight', format='png')

plt.show()

# Plotting of r-values containing p-value asterisk
f, ax = plt.subplots(figsize=(15, 19))
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

plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_asterisk_compact.eps'), dpi=1200,
            bbox_inches='tight', format='eps')
# plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_asterisk_full.eps'), dpi=1200,
#             bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_asterisk_compact.png'), dpi=1200,
            bbox_inches='tight', format='png')
# plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_asterisk_full.png'), dpi=1200,
#             bbox_inches='tight', format='png')

plt.show()
