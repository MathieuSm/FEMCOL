import numpy as np
import pandas as pd                                 # Used to manage data frames
import matplotlib.pyplot as plt                     # Used to perform plots
import os                                           # Used to manage path variables
from scipy import stats
import statsmodels.api as sm
import pylab

# Set directory & load data
Cwd = os.getcwd()
results_path = str(os.path.dirname(Cwd) + '/04_Results')
results_overview = str(os.path.dirname(Cwd) + '/04_Results/ResultsOverview.csv')
SavePath = results_path + '/04_Plots'
df = pd.read_csv(str(results_overview), skiprows=0)

pvalues = list()
for x in range(4, 44):
    statistic, pvalue = stats.ttest_ind(df[df['Gender'] == 'F'][df.columns[x]],
                                        df[df['Gender'] == 'M'][df.columns[x]], nan_policy='omit')
    pvalues.append(pvalue)
    plot_data_male = df.loc[(df['Gender'] == 'M'), [df.columns[x]]].dropna()
    plot_data_female = df.loc[(df['Gender'] == 'F'), [df.columns[x]]].dropna()
    male_qq = sm.qqplot(plot_data_male, line='s')
    female_qq = sm.qqplot(plot_data_female, line='s')
    pylab.close()
    # pylab.show()


result = pd.DataFrame(data=pvalues, index=df.columns[4:44])
