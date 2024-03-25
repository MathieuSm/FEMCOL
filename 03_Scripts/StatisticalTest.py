import numpy as np
import pandas as pd                                 # Used to manage data frames
import matplotlib.pyplot as plt                     # Used to perform plots
import os                                           # Used to manage path variables
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pylab
from pathlib import Path
import itertools as it


# Set directory & load data
test = Path.cwd()
Cwd = os.getcwd()
results_path = str(os.path.dirname(Cwd) + '/04_Results')
results_overview = str(os.path.dirname(Cwd) + '/04_Results/ResultsOverview.csv')
SavePath = results_path + '/04_Plots'
df = pd.read_csv(str(results_overview), skiprows=0)
df_new = df.rename(columns={'Modulus Mineralized / GPa': 'E_m',
                            'Modulus Demineralized / MPa': 'E_c',
                            'Ultimate Strain / -': 'USTRA',
                            'Ultimate Apparent Stress / MPa': 'USTRE',
                            'Tissue Mineral Density / mg HA / cm³': 'TMD',
                            'Mineral weight fraction / -': 'MWF',
                            'Organic weight fraction / -': 'OWF',
                            'Mineral to Matrix Ratio v2/a3 / -': 'MMR',
                            'Bone Volume Fraction / -': 'BVTV',
                            'Age / y': 'Age'})
pvalues = list()
statistic_age, pvalue_age = stats.ttest_ind(df[df['Gender'] == 'F']['Age / y'],
                                    df[df['Gender'] == 'M']['Age / y'], nan_policy='omit')
# for i, x in it.zip_longest(range(1, 2), range(4, 44)):
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

# Perform the ANCOVA
Em_Ec = ols('E_m ~ E_c + Gender + Gender * E_c', data=df_new).fit()
USTE_USTRA = ols('USTRE ~ USTRA + Gender + Gender * USTRA', data=df_new).fit()
USTRE_Ec = ols('USTRE ~ E_c + Gender + Gender * E_c', data=df_new).fit()
Ec_MWF = ols('E_c ~ MWF + Gender + Gender * MWF', data=df_new).fit()
Ec_TMD = ols('E_c ~ TMD + Gender + Gender * TMD', data=df_new).fit()

Em_Ec_age = ols('E_m ~ E_c + Gender + Age + Gender * Age', data=df_new).fit()
USTE_USTRA_age = ols('USTRE ~ USTRA + Gender + Age + Gender * Age', data=df_new).fit()
USTRE_Ec_age = ols('USTRE ~ E_c + Gender + Age + Gender * Age', data=df_new).fit()
Ec_MWF_age = ols('E_c ~ MWF + Gender + Age + Gender * Age', data=df_new).fit()
Ec_TMD_age = ols('E_c ~ TMD + Gender + Age + Gender * Age', data=df_new).fit()

# Print the summary of the model
print(Em_Ec.summary())
print(USTE_USTRA.summary())
print(USTRE_Ec.summary())
print(Ec_MWF.summary())
print(Ec_TMD.summary())

print(Em_Ec_age.summary())
print(USTE_USTRA_age.summary())
print(USTRE_Ec_age.summary())
print(Ec_MWF_age.summary())
print(Ec_TMD_age.summary())
