# This script plots various variables against each other, Data is retrieved from ResultsOverview.csv file

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
Cwd = os.getcwd()
results_path = str(os.path.dirname(Cwd) + '/04_Results')
results_overview = str(results_path + '/ResultsOverview.csv')
savepath_full = results_path + '/04_Plots/CorrelationMatrix_Full/'
savepath_compact = results_path + '/04_Plots/CorrelationMatrix_Compact/'

df = pd.read_csv(str(results_overview), skiprows=0)
df = df.drop(columns=['Sample ID', 'Site', 'Stiffness Mineralized / N/mm', 'Stiffness Demineralized / N/mm',
                      'Ultimate Force / N', 'Organic Weight / g', 'Mineral Weight / g', 'Water Weight / g',
                      'Bone Mineral Content / mg HA', 'Apparent Modulus Mineralized uFE / MPa', 'Yield Stress uFE / MPa',
                      'Ultimate Stress uFE / MPa', 'Haversian Canals Mean / %', 'Haversian Canals Std / %',
                      'Osteocytes Mean / %', 'Osteocytes Std / %', 'Cement Lines Mean / %', 'Cement Lines Std / %'])
df_new_all = df[['Age / y', 'Bone Volume Fraction / -', 'Tissue Mineral Density / mg HA / cm³',
                 'Mean Bone Area Fraction / -', 'Min Bone Area Fraction / -',
                 'Apparent Modulus Mineralized / GPa', 'Modulus Mineralized / GPa',
                 'Apparent Modulus Demineralized / MPa', 'Modulus Demineralized / MPa',
                 'Ultimate Apparent Stress / MPa', 'Ultimate Strain / -', 'Mineral weight fraction / -',
                 'Organic weight fraction / -', 'Water weight fraction / -', 'Mineral to Matrix Ratio v2/a3 / -',
                 'Crystallinity / -', 'Collagen dis/order / -', 'Matrix maturity / -']]
df_new_male = df[df['Gender'] == 'M'][['Age / y', 'Bone Volume Fraction / -', 'Tissue Mineral Density / mg HA / cm³',
                  'Mean Bone Area Fraction / -', 'Min Bone Area Fraction / -',
                  'Apparent Modulus Mineralized / GPa', 'Modulus Mineralized / GPa',
                  'Apparent Modulus Demineralized / MPa', 'Modulus Demineralized / MPa',
                  'Ultimate Apparent Stress / MPa', 'Ultimate Strain / -', 'Mineral weight fraction / -',
                  'Organic weight fraction / -', 'Water weight fraction / -', 'Mineral to Matrix Ratio v2/a3 / -',
                  'Crystallinity / -', 'Collagen dis/order / -', 'Matrix maturity / -']]
df_new_female = df[df['Gender'] == 'F'][['Age / y', 'Bone Volume Fraction / -', 'Tissue Mineral Density / mg HA / cm³',
                    'Mean Bone Area Fraction / -', 'Min Bone Area Fraction / -',
                    'Apparent Modulus Mineralized / GPa', 'Modulus Mineralized / GPa',
                    'Apparent Modulus Demineralized / MPa', 'Modulus Demineralized / MPa',
                    'Ultimate Apparent Stress / MPa', 'Ultimate Strain / -', 'Mineral weight fraction / -',
                    'Organic weight fraction / -', 'Water weight fraction / -', 'Mineral to Matrix Ratio v2/a3 / -',
                    'Crystallinity / -', 'Collagen dis/order / -', 'Matrix maturity / -']]

df_new_all.columns = ['Age', 'BV/TV', 'TMD', 'BA/TA_mean', 'BA/TA_min',
                      'AMM', 'MM', 'AMD', 'MD', 'UAPPSTRE', 'USTRA',
                      'WFM', 'WFO', 'WFW',
                      'MMRv2a3', 'XC', 'CDO', 'MMAT']
df_new_male.columns = ['Age', 'BV/TV', 'TMD', 'BA/TA_mean', 'BA/TA_min',
                      'AMM', 'MM', 'AMD', 'MD', 'UAPPSTRE', 'USTRA',
                      'WFM', 'WFO', 'WFW',
                      'MMRv2a3', 'XC', 'CDO', 'MMAT']
df_new_female.columns = ['Age', 'BV/TV', 'TMD', 'BA/TA_mean', 'BA/TA_min',
                      'AMM', 'MM', 'AMD', 'MD', 'UAPPSTRE', 'USTRA',
                      'WFM', 'WFO', 'WFW',
                      'MMRv2a3', 'XC', 'CDO', 'MMAT']

# # Correlation Matrix of complete dataset without differentiating gender
# Create empty dataframe for p-values and correlation matrix containing r values
corr_matrix_p = pd.DataFrame()
corr_matrix_r = round(df_new_all.corr(), 2)
test_matrix_r = pd.DataFrame()

# loop to iterate through rows and columns
for i in df_new_all.columns:
    for j in df_new_all.columns:

        # mask to select only values that are not NaN (linregress does not work otherwise)
        mask = ~df_new_all[i].isna() & ~df_new_all[j].isna()
        slope, intercept, r, pvalue, std_err = linregress(df_new_all[j][mask], df_new_all[i][mask])

        # create matrix containing p-values using mask criterion
        corr_matrix_p.loc[i, j] = pvalue

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

# Axis annotations of complete correlation matrix (need to be in the same order as in df_new)
# abbreviations = ['Age', 'BVTV', 'BMD', 'TMD', r'MMR$\nu_{2}a_{3}$', r'MMR$\nu_{1}a_{1}$', 'X$_c$', 'CDO', 'MMAT',
#                  'WF$_m$', 'WF$_o$', 'WF$_w$', r'$\rho_{b}$', 'E$_{app, m}$', 'E$_m$', 'E$_{app, c}$', 'E$_c$',
#                  '$\sigma_{app}$', '$\sigma_c$', '$\sigma_b$', '$\epsilon_c$', 'E$_{m, \mu FE}$', '$\sigma_{y, \mu FE}$',
#                  '$\sigma_{u, \mu FE}$', '$ECM_{AF_{mean}}$', '$ECM_{AF_{min}}$', '$ECM_{A_{CV}}$']

abbreviations = ['Age', r'$\rho$', 'TMD', 'BA/TA$_{mean}$', 'BA/TA$_{min}$',
                 'E$_{m}^{app}$', 'E$_m$', 'E$_{c}^{app}$', 'E$_c$', '$\sigma_{u}^{app}$', '$\epsilon_c$',
                 'WF$_m$', 'WF$_o$', 'WF$_w$',
                 r'MMR$\nu_{2}a_{3}$', 'X$_c$', 'CDO', 'MMAT']

# Font style and size
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "13"  # font size 10 for full corr. matrix, font size 13 for compact corr. matrix

# Added to "trick" the plot
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

# add column names as labels
ax.set_yticklabels(abbreviations, rotation=0, fontsize=14)
ax.set_xticklabels(abbreviations, fontsize=14)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_pvalues_compact_all.eps'), dpi=1200,
            bbox_inches='tight', format='eps')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_pvalues_full_all.eps'), dpi=1200,
#             bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_pvalues_compact_all.png'), dpi=1200,
            bbox_inches='tight', format='png')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_pvalues_full_all.png'), dpi=1200,
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

# add the column names as labels
ax.set_yticklabels(abbreviations, rotation=0, fontsize=14)
ax.set_xticklabels(abbreviations, fontsize=14)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_compact_all.eps'), dpi=1200,
            bbox_inches='tight', format='eps')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_rvalues_full_all.eps'), dpi=1200,
#             bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_compact_all.png'), dpi=1200,
            bbox_inches='tight', format='png')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_rvalues_full_all.png'), dpi=1200,
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

# add column names as labels
ax.set_yticklabels(abbreviations, rotation=0, fontsize=14)
ax.set_xticklabels(abbreviations, fontsize=14)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_asterisk_compact_all.eps'), dpi=1200,
            bbox_inches='tight', format='eps')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_rvalues_asterisk_full_all.eps'), dpi=1200,
#             bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_asterisk_compact_all.png'), dpi=1200,
            bbox_inches='tight', format='png')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_rvalues_asterisk_full_all.png'), dpi=1200,
#             bbox_inches='tight', format='png')
plt.show()

# # Correlation Matrix of female dataset
# Create empty dataframe for p-values and correlation matrix containing r values
corr_matrix_p = pd.DataFrame()
corr_matrix_r = round(df_new_female.corr(), 2)
test_matrix_r = pd.DataFrame()

# loop to iterate through rows and columns
for i in df_new_female.columns:
    for j in df_new_female.columns:

        # mask to select only values that are not NaN (linregress does not work otherwise)
        mask = ~df_new_female[i].isna() & ~df_new_female[j].isna()
        slope, intercept, r, pvalue, std_err = linregress(df_new_female[j][mask], df_new_female[i][mask])

        # create matrix containing p-values using mask criterion
        corr_matrix_p.loc[i, j] = pvalue

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

# Axis annotations of complete correlation matrix (need to be in the same order as in df_new)
# abbreviations = ['Age', 'BVTV', 'BMD', 'TMD', r'MMR$\nu_{2}a_{3}$', r'MMR$\nu_{1}a_{1}$', 'X$_c$', 'CDO', 'MMAT',
#                  'WF$_m$', 'WF$_o$', 'WF$_w$', r'$\rho_{b}$', 'E$_{app, m}$', 'E$_m$', 'E$_{app, c}$', 'E$_c$',
#                  '$\sigma_{app}$', '$\sigma_c$', '$\sigma_b$', '$\epsilon_c$', 'E$_{m, \mu FE}$', '$\sigma_{y, \mu FE}$',
#                  '$\sigma_{u, \mu FE}$', '$ECM_{AF_{mean}}$', '$ECM_{AF_{min}}$', '$ECM_{A_{CV}}$']

abbreviations = ['Age', r'$\rho$', 'TMD', 'BA/TA$_{mean}$', 'BA/TA$_{min}$',
                 'E$_{m}^{app}$', 'E$_m$', 'E$_{c}^{app}$', 'E$_c$', '$\sigma_{u}^{app}$', '$\epsilon_c$',
                 'WF$_m$', 'WF$_o$', 'WF$_w$',
                 r'MMR$\nu_{2}a_{3}$', 'X$_c$', 'CDO', 'MMAT']

# Font style and size
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "13"  # font size 10 for full corr. matrix, font size 13 for compact corr. matrix

# Added to "trick" the plot
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

# add column names as labels
ax.set_yticklabels(abbreviations, rotation=0, fontsize=14)
ax.set_xticklabels(abbreviations, fontsize=14)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_pvalues_compact_female.eps'), dpi=1200,
            bbox_inches='tight', format='eps')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_pvalues_full_female.eps'), dpi=1200,
#             bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_pvalues_compact_female.png'), dpi=1200,
            bbox_inches='tight', format='png')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_pvalues_full_female.png'), dpi=1200,
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

# add the column names as labels
ax.set_yticklabels(abbreviations, rotation=0, fontsize=14)
ax.set_xticklabels(abbreviations, fontsize=14)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_compact_female.eps'), dpi=1200,
            bbox_inches='tight', format='eps')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_rvalues_full_female.eps'), dpi=1200,
#             bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_compact_female.png'), dpi=1200,
            bbox_inches='tight', format='png')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_rvalues_full_female.png'), dpi=1200,
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

# add column names as labels
ax.set_yticklabels(abbreviations, rotation=0, fontsize=14)
ax.set_xticklabels(abbreviations, fontsize=14)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_asterisk_compact_female.eps'), dpi=1200,
            bbox_inches='tight', format='eps')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_rvalues_asterisk_full_female.eps'), dpi=1200,
#             bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_asterisk_compact_female.png'), dpi=1200,
            bbox_inches='tight', format='png')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_rvalues_asterisk_full_female.png'), dpi=1200,
#             bbox_inches='tight', format='png')
plt.show()

# # Correlation Matrix of male dataset
# Create empty dataframe for p-values and correlation matrix containing r values
corr_matrix_p = pd.DataFrame()
corr_matrix_r = round(df_new_male.corr(), 2)
test_matrix_r = pd.DataFrame()

# loop to iterate through rows and columns
for i in df_new_male.columns:
    for j in df_new_male.columns:

        # mask to select only values that are not NaN (linregress does not work otherwise)
        mask = ~df_new_male[i].isna() & ~df_new_male[j].isna()
        slope, intercept, r, pvalue, std_err = linregress(df_new_male[j][mask], df_new_male[i][mask])

        # create matrix containing p-values using mask criterion
        corr_matrix_p.loc[i, j] = pvalue

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

# Axis annotations of complete correlation matrix (need to be in the same order as in df_new)
# abbreviations = ['Age', 'BVTV', 'BMD', 'TMD', r'MMR$\nu_{2}a_{3}$', r'MMR$\nu_{1}a_{1}$', 'X$_c$', 'CDO', 'MMAT',
#                  'WF$_m$', 'WF$_o$', 'WF$_w$', r'$\rho_{b}$', 'E$_{app, m}$', 'E$_m$', 'E$_{app, c}$', 'E$_c$',
#                  '$\sigma_{app}$', '$\sigma_c$', '$\sigma_b$', '$\epsilon_c$', 'E$_{m, \mu FE}$', '$\sigma_{y, \mu FE}$',
#                  '$\sigma_{u, \mu FE}$', '$ECM_{AF_{mean}}$', '$ECM_{AF_{min}}$', '$ECM_{A_{CV}}$']

abbreviations = ['Age', r'$\rho$', 'TMD', 'BA/TA$_{mean}$', 'BA/TA$_{min}$',
                 'E$_{m}^{app}$', 'E$_m$', 'E$_{c}^{app}$', 'E$_c$', '$\sigma_{u}^{app}$', '$\epsilon_c$',
                 'WF$_m$', 'WF$_o$', 'WF$_w$',
                 r'MMR$\nu_{2}a_{3}$', 'X$_c$', 'CDO', 'MMAT']

# Font style and size
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "13"  # font size 10 for full corr. matrix, font size 13 for compact corr. matrix

# Added to "trick" the plot
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

# add column names as labels
ax.set_yticklabels(abbreviations, rotation=0, fontsize=14)
ax.set_xticklabels(abbreviations, fontsize=14)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_pvalues_compact_male.eps'), dpi=1200,
            bbox_inches='tight', format='eps')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_pvalues_full_male.eps'), dpi=1200,
#             bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_pvalues_compact_male.png'), dpi=1200,
            bbox_inches='tight', format='png')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_pvalues_full_male.png'), dpi=1200,
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

# add the column names as labels
ax.set_yticklabels(abbreviations, rotation=0, fontsize=14)
ax.set_xticklabels(abbreviations, fontsize=14)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_compact_male.eps'), dpi=1200,
            bbox_inches='tight', format='eps')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_rvalues_full_male.eps'), dpi=1200,
#             bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_compact_male.png'), dpi=1200,
            bbox_inches='tight', format='png')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_rvalues_full_male.png'), dpi=1200,
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

# add column names as labels
ax.set_yticklabels(abbreviations, rotation=0, fontsize=14)
ax.set_xticklabels(abbreviations, fontsize=14)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_asterisk_compact_male.eps'), dpi=1200,
            bbox_inches='tight', format='eps')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_rvalues_asterisk_full_male.eps'), dpi=1200,
#             bbox_inches='tight', format='eps')
plt.savefig(os.path.join(savepath_compact, 'correlation_matrix_heatmap_rvalues_asterisk_compact_male.png'), dpi=1200,
            bbox_inches='tight', format='png')
# plt.savefig(os.path.join(savepath_full, 'correlation_matrix_heatmap_rvalues_asterisk_full_male.png'), dpi=1200,
#             bbox_inches='tight', format='png')
plt.show()
