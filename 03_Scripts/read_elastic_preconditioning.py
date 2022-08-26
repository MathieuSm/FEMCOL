# This script plots various variables against each other, Data is retrieved from ResultsOverview.csv file

# Import standard packages
from pathlib import Path                 # Used to manage path variables in windows or linux

import matplotlib.image as mpl
import numpy as np                       # Used to do arrays (matrices) computations namely
import pandas as pd                      # Used to manage data frames
import matplotlib.pyplot as plt          # Used to perform plots
import statsmodels.formula.api as smf    # Used for statistical analysis (ols here)
import os
from scipy.stats import linregress  # Used to compute confidence intervals
import sys
import matplotlib.colors as col
import seaborn as sns
import math
import statistics


data = pd.DataFrame({'Cycles -': [1, 2, 3, 4, 5, 6, 7, 8],
                     'Stiffness N/mm': [6027, 5962, 5877, 5876, 5868, 5805, 5862, 5856]})
Cwd = Path.cwd()
savepath = Cwd / '04_Results/00_Mineralized'

reg = linregress(data['Cycles -'], data['Stiffness N/mm'])
slope = reg.slope
intercept = reg.intercept
y_hat = slope*data['Cycles -'] + intercept
p_value = round(reg.pvalue, 3)
rsquared_value = round(reg.rvalue * reg.rvalue, 3)
std_err = round(reg.stderr, 3)
mean = statistics.mean(data['Stiffness N/mm'])
std = statistics.stdev(data['Stiffness N/mm'])
t = 2.365
n = 8
upper_CI = round(mean + t * std/np.sqrt(n))
lower_CI = round(mean - t * std/np.sqrt(n))
Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=300, sharey=True, sharex=True)

plt.scatter(data['Cycles -'], data['Stiffness N/mm'], color='tab:blue')
plt.plot(data['Cycles -'], y_hat, color='darkorange')
plt.xlabel('Preconditioning cycles -')
plt.ylabel('Stiffness N/mm')

Axes.annotate(r'$R^2$ : ' + str(rsquared_value), xy=(0.05, 0.275), xycoords='axes fraction')
Axes.annotate(r'$SE$ : ' + str(std_err), xy=(0.05, 0.2), xycoords='axes fraction')
Axes.annotate(r'$p$ : ' + str(p_value), xy=(0.05, 0.125), xycoords='axes fraction')
Axes.annotate('95% CI [' + str(lower_CI) + r'$,$ ' + str(upper_CI) + ']', xy=(0.05, 0.05),
                          xycoords='axes fraction')
plt.autoscale()
plt.rcParams.update({'font.size': 14})
plt.savefig(os.path.join(savepath, '437R_Preconditioning.png'), dpi=300, bbox_inches='tight')
plt.show()



