import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path                            # Used to manage path variables in windows or linux
import numpy as np                                  # Used to do arrays (matrices) computations namely
import pandas as pd                                 # Used to manage data frames
import matplotlib.pyplot as plt                     # Used to perform plots
import statsmodels.formula.api as smf               # Used for statistical analysis (ols here)
import os                                           # Used to manage path variables
from scipy.stats.distributions import t             # Used to compute confidence intervals
import seaborn as sns                               # Used to create regression lines with confidence bands
import statistics                                   # Used to calculate statistical measures
from statsmodels.tools.eval_measures import rmse    # Used to evaluate rmse
from tqdm import tqdm


# Set directory & load data
Cwd = Path.cwd()
DataPath = Cwd / '04_Results/ResultsOverview.csv'
# DataPath = 'C:/Users/Stefan/PycharmProjects/FEMCOL/04_Results/ResultsOverviewMod.csv'
df = pd.read_csv(str(DataPath), skiprows=0)
df = df.drop(columns={'Min Equivalent Diameter / mm', 'Mean Apparent Diameter / mm'})
df = df.dropna()
df = df.reset_index(drop=True)

x = df[['Age / y', 'Mineral weight fraction / -']]
y = df[['Modulus Mineralized / MPa']]

x_abbrev = x.copy()
x_abbrev = x_abbrev.rename(columns={'Age / y': 'Age', 'Mineral weight fraction / -': 'mwf'})
y_abbrev = y.copy()
y_abbrev = y_abbrev.rename(columns={'Modulus Mineralized / MPa': 'Em'})
data = pd.DataFrame()
data['Em'] = y_abbrev['Em']
data['Age'] = x_abbrev['Age']
data['mwf'] = x_abbrev['mwf']


# with statsmodels
x = sm.add_constant(x) # adding a constant

model = sm.OLS(y, x).fit()
predictions = model.predict(x)

print_model = model.summary()
print(print_model)

mls_results = open('/home/stefan/PycharmProjects/FEMCOL/04_Results/04_Plots/mls_results.csv', 'w')
n = mls_results.write(print_model.as_csv())
mls_results.close()

model_interaction = smf.ols(formula='Em ~ Age + mwf + Age:mwf', data=data).fit()
summary_interaction = model_interaction.summary()
print(summary_interaction)

mls_results_interaction = open('/home/stefan/PycharmProjects/FEMCOL/04_Results/04_Plots/mls_results_interaction.csv', 'w')
n = mls_results_interaction.write(summary_interaction.as_csv())
mls_results_interaction.close()

X = sm.add_constant(data)
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
print(vif)
