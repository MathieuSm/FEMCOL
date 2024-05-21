# This script visualizes the age/sex distribution of the samples

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats.distributions import norm
import statsmodels.api as sm
import pylab

desired_width = 500
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', desired_width)
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width, suppress=True, formatter={'float_kind': '{:3}'.format})
plt.rc('font', size=12)

Cwd = os.getcwd()
SavePath = str(os.path.dirname(Cwd) + '/04_Results/04_Plots')
DataPath = str(os.path.dirname(Cwd) + '/04_Results/ResultsOverview.csv')

df = pd.read_csv(str(DataPath), skiprows=0)
df = df[df['Age / y'].notna()].reset_index(drop=True)

Data = pd.DataFrame()
Data['Age / y'] = df['Age / y']

# 01 Get data attributes
X = Data['Age / y']
SortedValues = np.sort(X.values)
N = len(X)
X_Bar = np.mean(X)
S_X = np.std(X, ddof=1)

# 05 Kernel density estimation (Gaussian kernel)
KernelEstimator = np.zeros(N)
NormalIQR = np.sum(np.abs(norm.ppf(np.array([0.25, 0.75]), 0, 1)))
DataIQR = np.abs(X.quantile(0.75)) - np.abs(X.quantile(0.25))
KernelHalfWidth = 0.9 * N ** (-1 / 5) * min(np.abs([S_X, DataIQR / NormalIQR]))
for Value in SortedValues:
    KernelEstimator += norm.pdf(SortedValues - Value, 0, KernelHalfWidth * 2)
KernelEstimator = KernelEstimator / N

## Histogram and density distribution
TheoreticalDistribution = norm.pdf(SortedValues, X_Bar, S_X)
Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=300)
Axes.hist(X, density=True, bins=20, edgecolor=(0, 0, 1), color=(1, 1, 1), label='Histogram')
Axes.plot(SortedValues, KernelEstimator, color=(1, 0, 0), label='Kernel Density')
Axes.plot(SortedValues, TheoreticalDistribution, linestyle='--', color=(0, 0, 0), label='Normal Distribution')
plt.xlabel('Donor Age')
plt.ylabel('Density (-)')
# plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15), prop={'size': 10})
plt.legend(loc='upper left')
plt.savefig(os.path.join(SavePath, 'AgeDistribution.eps'), dpi=300, bbox_inches='tight', format='eps')
plt.show()

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=300)
sm.qqplot(X, line='s')
plt.savefig(os.path.join(SavePath, 'AgeDistribution_qqplot.png'), dpi=300, bbox_inches='tight', format='png')
pylab.show()
