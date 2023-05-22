# This script plots various variables against each other, Data is retrieved from ResultsOverview.csv file

# Import standard packages
from pathlib import Path                            # Used to manage path variables in windows or linux
import numpy as np                                  # Used to do arrays (matrices) computations namely
import pandas as pd                                 # Used to manage data frames
import matplotlib.pyplot as plt                     # Used to perform plots
import statsmodels.formula.api as smf               # Used for statistical analysis (ols here)
import os                                           # Used to manage path variables
from scipy.stats.distributions import t             # Used to compute confidence intervals
import seaborn as sns                               # Used to create regression lines with confidence bands
import statistics as stats                          # Used to calculate statistical measures
from statsmodels.tools.eval_measures import rmse    # Used to evaluate rmse
from tqdm import tqdm                               # Used to track script progression while running


# Set directory & load data
Cwd = Path.cwd()
DataPath = Cwd / '04_Results/ResultsOverview.csv'
SavePath = Cwd / '04_Results/04_Plots'
# DataPath = 'C:/Users/Stefan/PycharmProjects/FEMCOL/04_Results/ResultsOverviewMod.csv'
df = pd.read_csv(str(DataPath), skiprows=0)
df = df.drop(columns={'Min Equivalent Diameter / mm', 'Mean Apparent Diameter / mm'})

E_m = df['Modulus Mineralized / MPa'].dropna().reset_index(drop=True)
E_app_m = df['Apparent Modulus Mineralized / MPa'].dropna().reset_index(drop=True)

E_c = df['Modulus Demineralized / MPa'].dropna().reset_index(drop=True)
E_app_c = df['Apparent Modulus Demineralized / MPa'].dropna().reset_index(drop=True)

k_m = df['Stiffness Mineralized / N/mm'].dropna().reset_index(drop=True)
k_c = df['Stiffness Demineralized N/mm'].dropna().reset_index(drop=True)

sigma_u = df['Ultimate Stress / MPa'].dropna().reset_index(drop=True)
sigma_u_app = df['Ultimate Apparent Stress / MPa'].dropna().reset_index(drop=True)
sigma_u_col = df['Ultimate Collagen Stress / MPa'].dropna().reset_index(drop=True)

f_u = df['Ultimate Force / N'].dropna().reset_index(drop=True)

epsilon_u = df['Ultimate Strain / -'].dropna().reset_index(drop=True)

d_b = df['Density / g/cm³'].dropna().reset_index(drop=True)

Wo = df['Organic Weight / g'].dropna().reset_index(drop=True)
Wm = df['Mineral Weight / g'].dropna().reset_index(drop=True)
Ww = df['Water weight / g'].dropna().reset_index(drop=True)

WFo = df['Organic Weight Fraction / -'].dropna().reset_index(drop=True)
WFm = df['Mineral Weight Fraction / -'].dropna().reset_index(drop=True)
WFw = df['Water Weight Fraction / -'].dropna().reset_index(drop=True)

BVTV = df['Bone Volume Fraction / -'].dropna().reset_index(drop=True)

BMD = df['Bone Mineral Density / mg HA / cm³']
TMD = df['Tissue Mineral Density / mg HA / cm³']

BMC = df['Bone Mineral Content / mg HA']



CVECMA = df['Coefficient of Variation / -']

MinECMA = df['Min ECM Area / ']
MEANAPPA = df['Mean Apparent Area / ']
MEANECMA = df['Mean ECM Area / ']

MEANECMAF = df['Mean ECM Area Fraction / -']
MINECMAF = df['Min ECM Area Fraction / -']

MMRV2A3 = df['Mineral to Matrix Ratio v2/a3 / -']
MMRV1A1 = df['Mineral to Matrix Ratio v1/a1 / -']

Xc = df['Crystallinity / -']

COLDIS = df['Collagen dis/order / -']

MATMAT = df['Matrix maturity / -']



columns = [E_m, E_app_m]
columns2 = []
columns3 = []

YposCI = 0.025
YposCV = YposCI + 0.075
YposN = YposCV + 0.075


# fist subplot
plt.figure()
plt.subplot(3, 3, 1)
box = plt.boxplot(columns, patch_artist=True, showmeans=True, meanline=True)
plt.tight_layout()
columns = [mm, amm]
plt.xticks([1, 2], ['E$_{app, m}$', 'E$_m$'])
colors = ['lightblue', 'lightgreen']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('MPa')

min = columns[0].min()
max = columns[0].max()
mean = stats.mean(columns[0])
median = stats.median(columns[0])
std = stats.stdev(columns[0])
cv = std/mean

# plt.annotate(r'$\downarrow$: ' + str(min) + r', $\uparrow$:'  + str(max), xy=(0.05, YposN), xycoords='axes fraction', fontsize=5)
# plt.annotate(r'$mean \pm std$ : ' + str(mean) + ', $\pm $' + str(round(std,1)), xy=(0.05, YposCV), xycoords='axes fraction', fontsize=5)
# plt.annotate(r'$CV$ : ' + str(round(cv, 3)), xy=(0.05, YposCI), xycoords='axes fraction', fontsize=5)

# second subplot
plt.subplot(3, 3, 2)
box = plt.boxplot(columns, patch_artist=True, showmeans=True, meanline=True)
plt.tight_layout()
columns2 = [mm, amm]
plt.xticks([1, 2], ['E$_{app, m}$', 'E$_m$'])
colors = ['lightblue', 'lightgreen']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('MPa')

min = columns[0].min()
max = columns[0].max()
mean = stats.mean(columns[0])
median = stats.median(columns[0])
std = stats.stdev(columns[0])
cv = std/mean

plt.savefig(os.path.join(SavePath, 'Moduli.png'), dpi=300, format='png')
plt.show()




