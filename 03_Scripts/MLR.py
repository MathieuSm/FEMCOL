import statsmodels.api as sm
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
y = df['Modulus Mineralized / MPa']

# with statsmodels
x = sm.add_constant(x) # adding a constant

model = sm.OLS(y, x).fit()
predictions = model.predict(x)

print_model = model.summary()
print(print_model)
