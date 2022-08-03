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

# Set directory & load data
Cwd = Path.cwd()
# DataPath = '/home/stefan/Documents/PythonScripts/04_Results/ResultsOverview.csv'
DataPath = 'C:/Users/Stefan/Dropbox/02_MScThesis/09_Results/ResultsOverview.csv'
df = pd.read_csv(str(DataPath), skiprows=0)
SampleID = df['Sample ID'].values.tolist()

# Create dataframe with variable names & respective abbreviations
ColumnNames = pd.DataFrame()
ColumnNames['Column Names'] = df.columns
column_names_abbrev = ['SID', 'Age', 'G', 'SM', 'AMM', 'UF', 'USTRE', 'USTRA', 'SD', 'AMD', 'D', 'OW', 'MW', 'WW',
                       'MWF', 'OWF', 'WWF', 'BVTV', 'BMD', 'TMD', 'BMC', 'MINA', 'MEANA', 'MINED', 'MEANED', 'MEANAF',
                       'MINAF', 'MAXAF']
ColumnNames['Abbreviations'] = column_names_abbrev
print(ColumnNames)

# Create plots by using indices of specific variable names
x_numbers = [20, 18, 18, 17, 17, 4, 22, 24, 15, 11, 11, 11, 26, 12, 12, 20, 20]
y_numbers = [3, 4, 6, 4, 6, 9, 9, 4, 6, 6, 3, 8, 6, 8, 3, 12, 14]

# Check if x_numbers and y_numbers are of equal length -> if not: abort script
if len(x_numbers) != len(y_numbers):
    print('\n ******** x_numbers and y_numbers do not have the same length ********')
    sys.exit(1)
else:
    # for loop to iterate through lists of numbers & create plots
    results = list()
    for i in range(len(x_numbers)):
        x_axis = ColumnNames['Column Names'][x_numbers[i]]
        y_axis = ColumnNames['Column Names'][y_numbers[i]]
        x_axis_abbrev = ColumnNames['Abbreviations'][x_numbers[i]]
        y_axis_abbrev = ColumnNames['Abbreviations'][y_numbers[i]]
        Data = df.filter(['Sample ID', x_axis, y_axis, 'Gender', 'Age']).dropna()
        Data2Fit = Data.copy()
        Data2Fit.rename(columns={'Sample ID': 'SID', x_axis: x_axis_abbrev, y_axis: y_axis_abbrev}, inplace=True)
        Data2Fit = Data2Fit.set_index('SID')
        FitResults = smf.ols(y_axis_abbrev + ' ~ 1 + ' + x_axis_abbrev, data=Data2Fit).fit()

        # PlotRegressionResults(FitResults, Data)

        # Calculate R^2, p-value, 95% CI, SE, N
        Y_Obs = FitResults.model.endog
        Y_Fit = FitResults.fittedvalues
        E = Y_Obs - Y_Fit
        RSS = np.sum(E ** 2)
        SE = np.sqrt(RSS / FitResults.df_resid)
        CI_l = FitResults.conf_int()[0][1]
        CI_r = FitResults.conf_int()[1][1]
        N = int(FitResults.nobs)
        R2 = FitResults.rsquared
        p = FitResults.pvalues[1]

        # Requirements for rounding: depending on value
        if abs(CI_l) >= 100:
            CI_l = round(CI_l)
        elif abs(CI_l) >= 1:
            CI_l = round(CI_l, 1)
        elif abs(CI_l) == 0:
            CI_l = int(CI_l)
        elif abs(CI_l) >= 0.001:
            CI_l = round(CI_l, 3)
        elif abs(CI_l) < 0.001:
            CI_l = '{:.2e}'.format(CI_l)

        if abs(CI_r) >= 100:
            CI_r = round(CI_r)
        elif abs(CI_r) >= 1:
            CI_r = round(CI_r, 1)
        elif abs(CI_r) == 0:
            CI_r = int(CI_r)
        elif abs(CI_r) >= 0.001:
            CI_r = round(CI_r, 3)
        elif abs(CI_r) < 0.001:
            CI_r = '{:.2e}'.format(CI_r)

        if abs(SE) >= 100:
            SE = round(SE)
        elif abs(SE) >= 1:
            SE = round(SE, 1)
        elif abs(SE) == 0:
            SE = int(SE)
        elif abs(SE) >= 0.001:
            SE = round(SE, 3)
        elif abs(SE) < 0.001:
            SE = '{:.1e}'.format(SE)

        if p < 0.001:
            p = '{:.1e}'.format(p)
        else:
            p = round(p, 3)

        # Put everything into growing list and convert to DataFrame that is saved as .csv file
        values = [x_axis, y_axis, p, SE, round(R2, 3), N, CI_l, CI_r]
        results.append(values)
        result_dir = pd.DataFrame(results, columns=['X-axis', 'Y-axis', 'p-value', '\u03C3\u2091\u209B\u209C',
                                                    'R\u00B2', 'N', 'lower bound 95% CI', 'upper bound 95% CI'])
        result_dir.to_csv(os.path.join('C:/Users/Stefan/Dropbox/02_MScThesis/09_Results/04_Plots/ResultsPlots.csv'), index=False)

        # print(FitResults.conf_int())
