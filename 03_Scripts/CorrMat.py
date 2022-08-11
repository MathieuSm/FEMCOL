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
             'Mineral weight fraction -', 'Organic weight fraction -', 'Water weight fraction -', 'Density g/cm³',
             'Apparent Modulus Mineralized MPa', 'Apparent Modulus Demineralized MPa', 'Ultimate Stress MPa',
             'Ultimate Strain -']]
df_new.columns = ['Bone Volume Fraction', 'Bone Mineral Density', 'Tissue Mineral Density', 'Mineral Weight Fraction',
                  'Organic Weight Fraction', 'Water Weight Fraction', 'Bone Density', 'Apparent Modulus Mineralized',
                  'Apparent Modulus Demineralized', 'Ultimate Stress', 'Ultimate Strain']
corr_matrix = df_new.corr()
mask = np.zeros_like(corr_matrix, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 15))
heatmap = sns.heatmap(corr_matrix,
                      mask=mask,
                      square=True,
                      linewidths=.5,
                      cmap='plasma_r',
                      cbar_kws={'shrink': .5,
                                'ticks': [-1, -.5, 0, 0.5, 1]},
                      vmin=-1,
                      vmax=1,
                      annot=True,
                      annot_kws={'size': 12})

#add the column names as labels
ax.set_yticklabels(corr_matrix.columns, rotation=0)
ax.set_xticklabels(corr_matrix.columns)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
plt.savefig(os.path.join(savepath, 'correlation_matrix_heatmap.png'), dpi=300, bbox_inches='tight')

plt.show()

