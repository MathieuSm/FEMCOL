# This script is used to perform an analysis of covariance on the variables that are used in the publication

import pandas as pd
import os
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pylab
from pathlib import Path


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
                            'Age / y': 'Age',
                            'Gender': 'Sex'})

# # Calculate sex specific p values for each variable
# statistic_age, pvalue_age = stats.ttest_ind(df[df['Gender'] == 'F']['Age / y'], df[df['Gender'] == 'M']['Age / y'],
#                                             nan_policy='omit')
# pvalues = list()
# for x in range(4, 44):
#     statistic, pvalue = stats.ttest_ind(df[df['Gender'] == 'F'][df.columns[x]],
#                                         df[df['Gender'] == 'M'][df.columns[x]], nan_policy='omit')
#     pvalues.append(pvalue)
#     plot_data_male = df.loc[(df['Gender'] == 'M'), [df.columns[x]]].dropna()
#     plot_data_female = df.loc[(df['Gender'] == 'F'), [df.columns[x]]].dropna()
#     male_qq = sm.qqplot(plot_data_male, line='s')
#     female_qq = sm.qqplot(plot_data_female, line='s')
#     pylab.close()
#     # pylab.show()

# result = pd.DataFrame(data=pvalues, index=df.columns[4:44])

# Perform the ANCOVA several times with and without combination terms
Em_Ec_age1 = ols('E_m ~ E_c + Sex', data=df_new).fit()
Em_Ec_age2 = ols('E_m ~ E_c + Age', data=df_new).fit()
Em_Ec_age3 = ols('E_m ~ E_c + Sex + Age', data=df_new).fit()
Em_Ec_age4 = ols('E_m ~ E_c + Sex + Age + Sex * E_c', data=df_new).fit()
Em_Ec_age5 = ols('E_m ~ E_c + Sex + Age + Age * E_c', data=df_new).fit()
Em_Ec_age6 = ols('E_m ~ E_c + Sex + Age + Age * E_c + Sex * E_c', data=df_new).fit()

USTRE_USTRA_age1 = ols('USTRE ~ USTRA + Sex', data=df_new).fit()
USTRE_USTRA_age2 = ols('USTRE ~ USTRA + Age', data=df_new).fit()
USTRE_USTRA_age3 = ols('USTRE ~ USTRA + Sex + Age', data=df_new).fit()
USTRE_USTRA_age4 = ols('USTRE ~ USTRA + Sex + Age + Sex * USTRA', data=df_new).fit()
USTRE_USTRA_age5 = ols('USTRE ~ USTRA + Sex + Age + Age * USTRA', data=df_new).fit()
USTRE_USTRA_age6 = ols('USTRE ~ USTRA + Sex + Age + Sex * USTRA + Age * USTRA', data=df_new).fit()

USTRE_Ec_age1 = ols('USTRE ~ E_c + Sex', data=df_new).fit()
USTRE_Ec_age2 = ols('USTRE ~ E_c + Age', data=df_new).fit()
USTRE_Ec_age3 = ols('USTRE ~ E_c + Sex + Age', data=df_new).fit()
USTRE_Ec_age4 = ols('USTRE ~ E_c + Sex + Age + Sex * E_c', data=df_new).fit()
USTRE_Ec_age5 = ols('USTRE ~ E_c + Sex + Age + Age * E_c', data=df_new).fit()
USTRE_Ec_age6 = ols('USTRE ~ E_c + Sex + Age + Sex * E_c + Age * E_c', data=df_new).fit()

Ec_MWF_age1 = ols('E_c ~ MWF + Sex', data=df_new).fit()
Ec_MWF_age2 = ols('E_c ~ MWF + Age', data=df_new).fit()
Ec_MWF_age3 = ols('E_c ~ MWF + Sex + Age', data=df_new).fit()
Ec_MWF_age4 = ols('E_c ~ MWF + Sex + Age + Sex * MWF', data=df_new).fit()
Ec_MWF_age5 = ols('E_c ~ MWF + Sex + Age + Age * MWF', data=df_new).fit()
Ec_MWF_age6 = ols('E_c ~ MWF + Sex + Age + Sex * MWF + Age * MWF', data=df_new).fit()

Ec_TMD_age1 = ols('E_c ~ TMD + Sex', data=df_new).fit()
Ec_TMD_age2 = ols('E_c ~ TMD + Age', data=df_new).fit()
Ec_TMD_age3 = ols('E_c ~ TMD + Sex + Age', data=df_new).fit()
Ec_TMD_age4 = ols('E_c ~ TMD + Sex + Age + Sex * TMD', data=df_new).fit()
Ec_TMD_age5 = ols('E_c ~ TMD + Sex + Age + Age * TMD', data=df_new).fit()
Ec_TMD_age6 = ols('E_c ~ TMD + Sex + Age + Sex * TMD + Age * TMD', data=df_new).fit()


# Print the summary of the model
print(Em_Ec_age1.summary())
print(Em_Ec_age2.summary())
print(Em_Ec_age3.summary())
print(Em_Ec_age4.summary())
print(Em_Ec_age5.summary())
print(Em_Ec_age6.summary())

print(USTE_USTRA_age1.summary())
print(USTE_USTRA_age2.summary())
print(USTE_USTRA_age3.summary())
print(USTE_USTRA_age4.summary())
print(USTE_USTRA_age5.summary())
print(USTE_USTRA_age6.summary())

print(USTRE_Ec_age1.summary())
print(USTRE_Ec_age2.summary())
print(USTRE_Ec_age3.summary())
print(USTRE_Ec_age4.summary())
print(USTRE_Ec_age5.summary())
print(USTRE_Ec_age6.summary())

print(Ec_MWF_age1.summary())
print(Ec_MWF_age2.summary())
print(Ec_MWF_age3.summary())
print(Ec_MWF_age4.summary())
print(Ec_MWF_age5.summary())
print(Ec_MWF_age6.summary())

print(Ec_TMD_age1.summary())
print(Ec_TMD_age2.summary())
print(Ec_TMD_age3.summary())
print(Ec_TMD_age4.summary())
print(Ec_TMD_age5.summary())
print(Ec_TMD_age6.summary())
