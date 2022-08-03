# This script plots various variables against each other, Data is retrieved from ResultsOverview.csv file

# Import standard packages
from pathlib import Path                 # Used to manage path variables in windows or linux

import matplotlib.image as mpl
import numpy as np                       # Used to do arrays (matrices) computations namely
import pandas as pd                      # Used to manage data frames
import matplotlib.pyplot as plt          # Used to perform plots
import statsmodels.formula.api as smf    # Used for statistical analysis (ols here)
import os
from scipy.stats.distributions import t  # Used to compute confidence intervals
import sys
import matplotlib.colors as col
import seaborn as sns
import math


# Set directory & load data
Cwd = Path.cwd()
DataPath = '/home/stefan/Documents/PythonScripts/04_Results/ResultsOverview.csv'
# DataPath = 'C:/Users/Stefan/Dropbox/02_MScThesis/09_Results/ResultsOverview.csv'
df = pd.read_csv(str(DataPath), skiprows=0)
SampleID = df['Sample ID'].values.tolist()

data = pd.DataFrame({'Age': df['Age'],
        'Ultimate Stress MPa': df['Ultimate Stress MPa']})
data = data.dropna().reset_index(drop=True)

sns.regplot(x=data['Age'], y=data['Ultimate Stress MPa'], color='0.7', scatter=False)
plt.scatter(data['Age'], data['Ultimate Stress MPa'])
plt.show()

sd = np.std(data['Ultimate Stress MPa'])
mean = np.mean(data['Ultimate Stress MPa'])
z = 1.96
n = len(data['Ultimate Stress MPa'])

CI_l = mean - z * sd / math.sqrt(n)
CI_r = mean + z * sd / math.sqrt(n)

print('CI_l = ' + str(CI_l))
print('CI_r = ' + str(CI_r))

