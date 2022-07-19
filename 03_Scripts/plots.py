# Import standard packages
from pathlib import Path                # Used to manage path variables in windows or linux
import numpy as np                      # Used to do arrays (matrices) computations namely
import pandas as pd                     # Used to manage data frames
import matplotlib.pyplot as plt         # Used to perform plots
import statsmodels.formula.api as smf   # Used for statistical analysis (ols here)
import os
from scipy.stats.distributions import t # Used to compute confidence intervals


# Set directories
Cwd = Path.cwd()
DataPath = Cwd / '04_Results/ResultsOverview.csv/'
df = pd.read_csv(str(DataPath), skiprows=0)


# Define some functions
def PlotRegressionResults(Model, Data, Alpha=0.95):

    print(Model.summary())

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

    ## Plots
    DPI = 300
    savepath = Cwd / '04_Results/'
    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=DPI, sharey=True, sharex=True)
    Axes.plot(X[:, 1], Y_Fit, color=(1, 0, 0), label='Fit')
    # Axes.fill_between(X_Obs, np.sort(CI_Line_o), np.sort(CI_Line_u), color=(0, 0, 0), alpha=0.1, label=str(int(
    #     Alpha*100)) + '% CI')
    Axes.plot(X[:, 1], Y_Obs, linestyle='none', marker='o', color=(0, 0, 0), fillstyle='none')
    Axes.annotate(r'$N$  : ' + str(N), xy=(0.05, 0.25), xycoords='axes fraction')
    Axes.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.05, 0.175), xycoords='axes fraction')
    Axes.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.05, 0.1), xycoords='axes fraction')
    Axes.set_ylabel(Data.columns[2])
    Axes.set_xlabel(Data.columns[1])
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.legend()
    plt.savefig(os.path.join(savepath, Data2Fit.columns[0] + '_' + Data2Fit.columns[1] + '.png'), dpi=300)
    plt.show()
    plt.close(Figure)


SampleID = df['Sample ID'].values.tolist()

# Build dataframe with age and stiffness mineralized
x_axis = 'Age'
y_axis = 'Stiffness Mineralized N/mm'
x_axis_abbrev = 'Age'
y_axis_abbrev = 'SM'
Data = df.filter(['Sample ID', x_axis, y_axis]).dropna()
Data2Fit = Data.copy()
Data2Fit.rename(columns={'Sample ID': 'SID', x_axis: x_axis_abbrev, y_axis: y_axis_abbrev}, inplace=True)
Data2Fit = Data2Fit.set_index('SID')

FitResults = smf.ols('SM ~ 1 + Age', data=Data2Fit).fit()
PlotRegressionResults(FitResults, Data)

print(FitResults.conf_int())

# Build dataframe with age and stiffness mineralized
x_axis = 'Age'
y_axis = 'Apparent Modulus Mineralized MPa'
x_axis_abbrev = 'Age'
y_axis_abbrev = 'AMM'
Data = df.filter(['Sample ID', x_axis, y_axis]).dropna()
Data2Fit = Data.copy()
Data2Fit.rename(columns={'Sample ID': 'SID', x_axis: x_axis_abbrev, y_axis: y_axis_abbrev}, inplace=True)
Data2Fit = Data2Fit.set_index('SID')

FitResults = smf.ols('AMM ~ 1 + Age', data=Data2Fit).fit()
PlotRegressionResults(FitResults, Data)

print(FitResults.conf_int())

# Build dataframe with age and stiffness mineralized
x_axis = 'Age'
y_axis = 'Ultimate Force N'
x_axis_abbrev = 'Age'
y_axis_abbrev = 'UF'
Data = df.filter(['Sample ID', x_axis, y_axis]).dropna()
Data2Fit = Data.copy()
Data2Fit.rename(columns={'Sample ID': 'SID', x_axis: x_axis_abbrev, y_axis: y_axis_abbrev}, inplace=True)
Data2Fit = Data2Fit.set_index('SID')

FitResults = smf.ols('UF ~ 1 + Age', data=Data2Fit).fit()
PlotRegressionResults(FitResults, Data)

print(FitResults.conf_int())

# Build dataframe with age and stiffness mineralized
x_axis = 'Age'
y_axis = 'Ultimate Stress MPa'
x_axis_abbrev = 'Age'
y_axis_abbrev = 'US'
Data = df.filter(['Sample ID', x_axis, y_axis]).dropna()
Data2Fit = Data.copy()
Data2Fit.rename(columns={'Sample ID': 'SID', x_axis: x_axis_abbrev, y_axis: y_axis_abbrev}, inplace=True)
Data2Fit = Data2Fit.set_index('SID')

FitResults = smf.ols('US ~ 1 + Age', data=Data2Fit).fit()
PlotRegressionResults(FitResults, Data)

print(FitResults.conf_int())

# Build dataframe with age and stiffness mineralized
x_axis = 'Age'
y_axis = 'Ultimate Strain -'
x_axis_abbrev = 'Age'
y_axis_abbrev = 'US'
Data = df.filter(['Sample ID', x_axis, y_axis]).dropna()
Data2Fit = Data.copy()
Data2Fit.rename(columns={'Sample ID': 'SID', x_axis: x_axis_abbrev, y_axis: y_axis_abbrev}, inplace=True)
Data2Fit = Data2Fit.set_index('SID')

FitResults = smf.ols('US ~ 1 + Age', data=Data2Fit).fit()
PlotRegressionResults(FitResults, Data)

print(FitResults.conf_int())

# Build dataframe with age and stiffness mineralized
x_axis = 'Age'
y_axis = 'Stiffness Demineralized Unloading N/mm'
x_axis_abbrev = 'Age'
y_axis_abbrev = 'SDU'
Data = df.filter(['Sample ID', x_axis, y_axis]).dropna()
Data2Fit = Data.copy()
Data2Fit.rename(columns={'Sample ID': 'SID', x_axis: x_axis_abbrev, y_axis: y_axis_abbrev}, inplace=True)
Data2Fit = Data2Fit.set_index('SID')

FitResults = smf.ols('SDU ~ 1 + Age', data=Data2Fit).fit()
PlotRegressionResults(FitResults, Data)

print(FitResults.conf_int())