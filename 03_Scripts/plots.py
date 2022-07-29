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

# Define some functions
def PlotRegressionResults(Model, Data, Alpha=0.95):

    # print(Model.summary())

    ## Plot results
    Y_Obs = Model.model.endog
    Y_Fit = Model.fittedvalues
    N = int(Model.nobs)
    C = np.matrix(Model.cov_params())
    X = np.matrix(Model.model.exog)
    X_Obs = np.sort(np.array(X[:, 1]).reshape(len(X)))

    ## Compute R2 and standard error of the estimate
    E = Y_Obs - Y_Fit
    RSS = np.sum(E ** 2)
    SE = np.sqrt(RSS / Model.df_resid)
    TSS = np.sum((Model.model.endog - Model.model.endog.mean()) ** 2)
    RegSS = TSS - RSS
    R2 = RegSS / TSS

    ## Compute CI lines
    B_0 = np.sqrt(np.diag(np.abs(X * C * X.T)))
    t_Alpha = t.interval(Alpha, N - X.shape[1] - 1)
    CI_Line_u = Y_Fit + t_Alpha[0] * SE * B_0
    CI_Line_o = Y_Fit + t_Alpha[1] * SE * B_0
    CI_l = FitResults.conf_int()[0][1]
    CI_r = FitResults.conf_int()[1][1]

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

    ## Plots
    DPI = 300
    savepath = Cwd / '04_Results/04_Plots/'
    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=DPI, sharey=True, sharex=True)
    Axes.plot(X[:, 1], Y_Fit, color=(1, 0, 0), label='Fit')
    # Axes.fill_between(X_Obs, np.sort(CI_Line_o), np.sort(CI_Line_u), color=(0, 0, 0), alpha=0.1, label=str(int(
    #     Alpha*100)) + '% CI')
    Axes.plot(X[:, 1][Data['Gender'] == 'M'], Y_Obs[Data['Gender'] == 'M'], linestyle='none', marker='o',
              color=(0, 0, 1), fillstyle='none', label='male')
    Axes.plot(X[:, 1][Data['Gender'] == 'F'], Y_Obs[Data['Gender'] == 'F'], linestyle='none', marker='x',
              color=(0, 0, 1), fillstyle='none', label='female')
    Axes.annotate(r'$N$  : ' + str(N), xy=(0.05, 0.325), xycoords='axes fraction')
    Axes.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.05, 0.25), xycoords='axes fraction')
    Axes.annotate(r'$\sigma_{est}$ : ' + str(SE), xy=(0.05, 0.175), xycoords='axes fraction')
    Axes.annotate(r'$p$ : ' + format(round(FitResults.pvalues[1], 3), '.3f'), xy=(0.05, 0.1), xycoords='axes fraction')
    Axes.annotate('95% CI [' + str(CI_l) + r'$,$ ' + str(CI_r) + ']', xy=(0.05, 0.025), xycoords='axes fraction')
    Axes.set_ylabel(Data.columns[2])
    Axes.set_xlabel(Data.columns[1])
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(savepath, Data2Fit.columns[0] + '_' + Data2Fit.columns[1] + '.png'), dpi=300)
    plt.show()
    plt.close(Figure)

# Set directory & load data
Cwd = Path.cwd()
DataPath = Cwd / '04_Results/ResultsOverview.csv/'
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
x_numbers = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 20, 18, 18, 17, 17, 4, 22, 24, 15, 11, 11, 11, 26,
             12, 12]
y_numbers = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 3, 4, 6, 4, 6, 9, 9, 4, 6, 6, 3, 8, 6, 8,
             3]

# Check if x_numbers and y_numbers are of equal length -> if not: abort script
if len(x_numbers) != len(y_numbers):
    print('\n ******** x_numbers and y_numbers do not have the same length ********')
    sys.exit(1)
else:
    # for loop to iterate through lists of numbers & create plots
    results = list()
    for i in range(len(x_numbers)-1):
        x_axis = ColumnNames['Column Names'][x_numbers[i]]
        y_axis = ColumnNames['Column Names'][y_numbers[i]]
        x_axis_abbrev = ColumnNames['Abbreviations'][x_numbers[i]]
        y_axis_abbrev = ColumnNames['Abbreviations'][y_numbers[i]]
        Data = df.filter(['Sample ID', x_axis, y_axis, 'Gender']).dropna()
        Data2Fit = Data.copy()
        Data2Fit.rename(columns={'Sample ID': 'SID', x_axis: x_axis_abbrev, y_axis: y_axis_abbrev}, inplace=True)
        Data2Fit = Data2Fit.set_index('SID')
        FitResults = smf.ols(y_axis_abbrev + ' ~ 1 + ' + x_axis_abbrev, data=Data2Fit).fit()
        PlotRegressionResults(FitResults, Data)

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

        # Put everything into growing list and convert to DataFrame that is saved as .csv file
        values = [x_axis, y_axis, round(p, 4), SE, round(R2, 3), N, CI_l, CI_r]
        results.append(values)
        result_dir = pd.DataFrame(results, columns=['X-axis', 'Y-axis', 'p-value', '\u03C3\u2091\u209B\u209C',
                                                    'R\u00B2', 'N', 'lower bound 95% CI', 'upper bound 95% CI'])
        result_dir.to_csv(os.path.join('/home/stefan/Documents/PythonScripts/04_Results/04_Plots/',
                                              'ResultsPlots.csv'), index=False)

        # print(FitResults.conf_int())

