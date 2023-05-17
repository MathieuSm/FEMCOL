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
import statistics                                   # Used to calculate statistical measures
from statsmodels.tools.eval_measures import rmse    # Used to evaluate rmse
from tqdm import tqdm                               # Used to track script progression while running


# Set directory & load data
Cwd = Path.cwd()
DataPath = Cwd / '04_Results/ResultsOverview.csv'
# DataPath = 'C:/Users/Stefan/PycharmProjects/FEMCOL/04_Results/ResultsOverviewMod.csv'
df = pd.read_csv(str(DataPath), skiprows=0)
df = df.drop(columns={'Min Equivalent Diameter / mm', 'Mean Apparent Diameter / mm'})


y_axis_name = df.columns[4]
y_axis = df[y_axis_name].dropna()
y_axis_index = y_axis.index

sampleID = df['Sample ID'][y_axis.index]


plt.scatter(sampleID, y_axis)
plt.ylabel(y_axis_name)
plt.xticks(sampleID, rotation = 90)
plt.show()
