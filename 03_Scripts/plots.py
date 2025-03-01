# This script plots various variables against each other and performs simple linear regression, Data is retrieved from
# ResultsOverview.csv file

# Import standard packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import os
from scipy.stats.distributions import t
import seaborn as sns
import statistics
from statsmodels.tools.eval_measures import rmse
from tqdm import tqdm
import matplotlib.ticker as ticker

# Set directory & load data
Cwd = os.getcwd()
Results_path = str(os.path.dirname(Cwd) + '/04_Results')
results_overview = str(os.path.dirname(Cwd) + '/04_Results/ResultsOverview.csv')
Savepath = Results_path + '/04_Plots/Individual_old'

isExist = os.path.exists(Savepath)
if not isExist:
    os.makedirs(Savepath)

df = pd.read_csv(str(results_overview), skiprows=0)
SampleID = df['Sample ID'].values.tolist()

# Create dataframe with variable names & respective abbreviations
ColumnNames = pd.DataFrame()
ColumnNames['Column Names'] = df.columns

AxisLabels = ColumnNames.replace(
    {'Apparent Modulus Mineralized / GPa': 'Apparent Modulus Mineralized E$_{m}^{app}$ / GPa',
     'Modulus Mineralized / GPa': 'Modulus Mineralized E$_{m}$ / GPa',
     'Ultimate Apparent Stress / MPa': 'Ultimate Apparent Stress $\sigma_{u}^{app}$ / MPa',
     'Ultimate Collagen Stress / MPa': 'Ultimate Collagen Stress $\sigma_{u}^{c}$ / MPa',
     'Ultimate Stress / MPa': 'Ultimate Stress $\sigma_{u}^{b}$ / MPa',
     'Coefficient of Variation / -': 'Coefficient of Variation CV / -',
     'Ultimate Strain / -': 'Ultimate Strain $\u03f5_{u}$',
     'Apparent Modulus Demineralized / MPa': 'Apparent Modulus Demineralized E$_{o}^{app}$ / MPa',
     'Modulus Demineralized / MPa': 'Modulus Demineralized E$_{o}$ / MPa',
     'Density / g/cm³': 'Density ' + r'$\rho_{b}$ / g / cm³',
     'Organic Weight / g': 'Organic Weight m$_{o}$ / g',
     'Mineral Weight / g': 'Mineral Weight m$_{m}$ / g',
     'Water Weight / g': 'Water Weight m$_{w}$ / g',
     'Mineral weight fraction / -': 'Mineral Weight Fraction WF$_{m}$ / -',
     'Organic weight fraction / -': 'Organic Weight Fraction WF$_{o}$ / -',
     'Water weight fraction / -': 'Water Weight Fraction WF$_{w}$ / -',
     'Bone Volume Fraction / -': 'Bone Volume Fraction ' + r'$\rho$ / -',
     'Bone Mineral Density / mg HA / cm³': 'Bone Mineral Density BMD / mg / cm³',
     'Tissue Mineral Density / mg HA / cm³': 'Tissue Mineral Density TMD / mg / cm³',
     'Bone Mineral Content / mg HA': 'Bone Mineral Content BMC / mg HA',
     'Min ECM Area / mm²': 'Min ECM Area A$_{F}^{min}$ / mm²',
     'Mean Apparent Area / mm²': 'Mean Apparent Area A$_{app}^{mean}$ / mm²',
     'Mean ECM Area / mm²': 'Mean ECM Area A$_{F}^{mean}$ / mm²',
     'Mean ECM Area Fraction / -': 'Mean ECM Area Fraction BA/TA$_{mean}$ / -',
     'Min ECM Area Fraction / -': 'Min ECM Area Fraction BA/TA$_{min}$ / -',
     'Mineral to Matrix Ratio v2/a3 / -': r'Mineral-to-Matrix Ratio ' r'$\rho_{MM}$ ' + r'$\nu_{2}/a_{3}$ / -',
     'Mineral to Matrix Ratio v1/a1 / -': r'Mineral-to-Matrix Ratio ' r'$\rho_{MM}$ ' + r'$\nu_{1}/a_{1}$ / -',
     'Crystallinity / -': 'Crystallinity X$_{c}$ / -',
     'Collagen dis/order / -': 'Collagen dis/order / -',
     'Matrix maturity / -': 'Matrix maturity / -',
     'Relative Pyridinoline Content / -': 'Relative Pyridinoline Content / -',
     'Relative Proteoglycan Content / -': 'Relative Proteoglycan Content / -',
     'Relative Lipid Content / -': 'Relative Lipid Content / -',
     'Apparent Modulus Mineralized uFE / MPa': 'Apparent Modulus Mineralized $\mu$FE E$^{\mu FE}_{app, m}$ / MPa',
     'Yield Stress uFE / MPa': 'Yield Stress $\mu$FE $\sigma^{\mu FE}_{y}$ / MPa',
     'Ultimate Stress uFE / MPa': 'Ultimate Stress $\mu$FE $\sigma^{\mu FE}_{app}$ / MPa'})

column_names_abbrev = ['SID', 'Age', 'G', 'Site', 'SM', 'SD', 'EAPPM', 'EM', 'UF', 'UAPPSTRE', 'UCSTRE', 'USTRE',
                       'USTRA', 'EAPPO', 'EO', 'D', 'OW', 'MW', 'WW', 'WFM', 'WFO', 'WFW', 'BVTV', 'BMD', 'TMD', 'BMC',
                       'ABMean', 'ABMin', 'MMRv2a3', 'MMRv1a1', 'CRY', 'COLDIS', 'MATMAT', 'RPyC', 'RProC', 'RLC',
                       'EAPPFE', 'YSTREFE', 'USTREFE', 'HCMean', 'HCStd', 'OCMean', 'OCStd', 'CLMean', 'CLStd']
ColumnNames['Abbreviations'] = column_names_abbrev
AxisLabels['Abbreviations'] = column_names_abbrev

# Pairs of properties that are regressed against each other; has the Format [x-axis, y-axis] (stress & moduli should
# be on y-axis for non-age plots)
Pair = pd.DataFrame([
    ['Age / y', 'Apparent Modulus Demineralized / MPa'],
    ['Age / y', 'Apparent Modulus Mineralized / GPa'],
    ['Age / y', 'Modulus Mineralized / GPa'],
    ['Age / y', 'Modulus Demineralized / MPa'],
    ['Age / y', 'Bone Mineral Content / mg HA'],
    ['Age / y', 'Bone Mineral Density / mg HA / cm³'],
    ['Age / y', 'Bone Volume Fraction / -'],
    ['Age / y', 'Density / g/cm³'],
    ['Age / y', 'Mineral Weight / g'],
    ['Age / y', 'Mineral weight fraction / -'],
    ['Age / y', 'Organic Weight / g'],
    ['Age / y', 'Organic weight fraction / -'],
    ['Age / y', 'Stiffness Demineralized / N/mm'],
    ['Age / y', 'Stiffness Mineralized / N/mm'],
    ['Age / y', 'Tissue Mineral Density / mg HA / cm³'],
    ['Age / y', 'Ultimate Force / N'],
    ['Age / y', 'Ultimate Strain / -'],
    ['Age / y', 'Ultimate Apparent Stress / MPa'],
    ['Age / y', 'Ultimate Collagen Stress / MPa'],
    ['Age / y', 'Ultimate Stress / MPa'],
    ['Age / y', 'Water Weight / g'],
    ['Age / y', 'Water weight fraction / -'],
    ['Age / y', 'Mineral to Matrix Ratio v2/a3 / -'],
    ['Age / y', 'Mineral to Matrix Ratio v1/a1 / -'],
    ['Age / y', 'Crystallinity / -'],
    ['Age / y', 'Collagen dis/order / -'],
    ['Age / y', 'Matrix maturity / -'],
    ['Age / y', 'Relative Pyridinoline Content / -'],
    ['Age / y', 'Relative Proteoglycan Content / -'],
    ['Age / y', 'Relative Lipid Content / -'],
    ['Apparent Modulus Mineralized / GPa', 'Modulus Mineralized / GPa'],
    ['Apparent Modulus Mineralized / GPa', 'Ultimate Apparent Stress / MPa'],
    ['Apparent Modulus Mineralized / GPa', 'Ultimate Collagen Stress / MPa'],
    ['Apparent Modulus Mineralized / GPa', 'Ultimate Stress / MPa'],
    ['Apparent Modulus Mineralized / GPa', 'Ultimate Strain / -'],
    ['Apparent Modulus Mineralized / GPa', 'Apparent Modulus Demineralized / MPa'],
    ['Apparent Modulus Mineralized / GPa', 'Modulus Demineralized / MPa'],
    ['Apparent Modulus Mineralized / GPa', 'Density / g/cm³'],
    ['Apparent Modulus Mineralized / GPa', 'Mineral weight fraction / -'],
    ['Apparent Modulus Mineralized / GPa', 'Organic weight fraction / -'],
    ['Apparent Modulus Mineralized / GPa', 'Water weight fraction / -'],
    ['Apparent Modulus Mineralized / GPa', 'Bone Volume Fraction / -'],
    ['Apparent Modulus Mineralized / GPa', 'Bone Mineral Density / mg HA / cm³'],
    ['Apparent Modulus Mineralized / GPa', 'Tissue Mineral Density / mg HA / cm³'],
    ['Apparent Modulus Mineralized / GPa', 'Mineral to Matrix Ratio v2/a3 / -'],
    ['Apparent Modulus Mineralized / GPa', 'Mineral to Matrix Ratio v1/a1 / -'],
    ['Apparent Modulus Mineralized / GPa', 'Crystallinity / -'],
    ['Apparent Modulus Mineralized / GPa', 'Collagen dis/order / -'],
    ['Apparent Modulus Mineralized / GPa', 'Matrix maturity / -'],
    ['Apparent Modulus Mineralized / GPa', 'Relative Pyridinoline Content / -'],
    ['Apparent Modulus Mineralized / GPa', 'Relative Proteoglycan Content / -'],
    ['Apparent Modulus Mineralized / GPa', 'Relative Lipid Content / -'],
    ['Ultimate Apparent Stress / MPa', 'Modulus Mineralized / GPa'],
    ['Ultimate Collagen Stress / MPa', 'Modulus Mineralized / GPa'],
    ['Ultimate Stress / MPa', 'Modulus Mineralized / GPa'],
    ['Ultimate Strain / -', 'Modulus Mineralized / GPa'],
    ['Apparent Modulus Demineralized / MPa', 'Modulus Mineralized / GPa'],
    ['Modulus Demineralized / MPa', 'Modulus Mineralized / GPa'],
    ['Density / g/cm³', 'Modulus Mineralized / GPa'],
    ['Mineral weight fraction / -', 'Modulus Mineralized / GPa'],
    ['Organic weight fraction / -', 'Modulus Mineralized / GPa'],
    ['Water weight fraction / -', 'Modulus Mineralized / GPa'],
    ['Bone Volume Fraction / -', 'Modulus Mineralized / GPa'],
    ['Bone Mineral Density / mg HA / cm³', 'Modulus Mineralized / GPa'],
    ['Tissue Mineral Density / mg HA / cm³', 'Modulus Mineralized / GPa'],
    ['Mineral to Matrix Ratio v2/a3 / -', 'Modulus Mineralized / GPa'],
    ['Mineral to Matrix Ratio v1/a1 / -', 'Modulus Mineralized / GPa'],
    ['Crystallinity / -', 'Modulus Mineralized / GPa'],
    ['Collagen dis/order / -', 'Modulus Mineralized / GPa'],
    ['Matrix maturity / -', 'Modulus Mineralized / GPa'],
    ['Relative Pyridinoline Content / -', 'Modulus Mineralized / GPa'],
    ['Relative Proteoglycan Content / -', 'Modulus Mineralized / GPa'],
    ['Relative Lipid Content / -', 'Modulus Mineralized / GPa'],
    ['Ultimate Collagen Stress / MPa', 'Ultimate Apparent Stress / MPa'],
    ['Ultimate Stress / MPa', 'Ultimate Apparent Stress / MPa'],
    ['Ultimate Strain / -', 'Ultimate Apparent Stress / MPa'],
    ['Apparent Modulus Demineralized / MPa', 'Ultimate Apparent Stress / MPa'],
    ['Modulus Demineralized / MPa', 'Ultimate Apparent Stress / MPa'],
    ['Density / g/cm³', 'Ultimate Apparent Stress / MPa'],
    ['Mineral weight fraction / -', 'Ultimate Apparent Stress / MPa'],
    ['Organic weight fraction / -', 'Ultimate Apparent Stress / MPa'],
    ['Water weight fraction / -', 'Ultimate Apparent Stress / MPa'],
    ['Bone Volume Fraction / -', 'Ultimate Apparent Stress / MPa'],
    ['Bone Mineral Density / mg HA / cm³', 'Ultimate Apparent Stress / MPa'],
    ['Tissue Mineral Density / mg HA / cm³', 'Ultimate Apparent Stress / MPa'],
    ['Mineral to Matrix Ratio v2/a3 / -', 'Ultimate Apparent Stress / MPa'],
    ['Mineral to Matrix Ratio v1/a1 / -', 'Ultimate Apparent Stress / MPa'],
    ['Crystallinity / -', 'Ultimate Apparent Stress / MPa'],
    ['Collagen dis/order / -', 'Ultimate Apparent Stress / MPa'],
    ['Matrix maturity / -', 'Ultimate Apparent Stress / MPa'],
    ['Relative Pyridinoline Content / -', 'Ultimate Apparent Stress / MPa'],
    ['Relative Proteoglycan Content / -', 'Ultimate Apparent Stress / MPa'],
    ['Relative Lipid Content / -', 'Ultimate Apparent Stress / MPa'],
    ['Ultimate Stress / MPa', 'Ultimate Collagen Stress / MPa'],
    ['Ultimate Strain / -', 'Ultimate Collagen Stress / MPa'],
    ['Apparent Modulus Demineralized / MPa', 'Ultimate Collagen Stress / MPa'],
    ['Modulus Demineralized / MPa', 'Ultimate Collagen Stress / MPa'],
    ['Density / g/cm³', 'Ultimate Collagen Stress / MPa'],
    ['Mineral weight fraction / -', 'Ultimate Collagen Stress / MPa'],
    ['Organic weight fraction / -', 'Ultimate Collagen Stress / MPa'],
    ['Water weight fraction / -', 'Ultimate Collagen Stress / MPa'],
    ['Bone Volume Fraction / -', 'Ultimate Collagen Stress / MPa'],
    ['Bone Mineral Density / mg HA / cm³', 'Ultimate Collagen Stress / MPa'],
    ['Tissue Mineral Density / mg HA / cm³', 'Ultimate Collagen Stress / MPa'],
    ['Mineral to Matrix Ratio v2/a3 / -', 'Ultimate Collagen Stress / MPa'],
    ['Mineral to Matrix Ratio v1/a1 / -', 'Ultimate Collagen Stress / MPa'],
    ['Crystallinity / -', 'Ultimate Collagen Stress / MPa'],
    ['Collagen dis/order / -', 'Ultimate Collagen Stress / MPa'],
    ['Matrix maturity / -', 'Ultimate Collagen Stress / MPa'],
    ['Relative Pyridinoline Content / -', 'Ultimate Collagen Stress / MPa'],
    ['Relative Proteoglycan Content / -', 'Ultimate Collagen Stress / MPa'],
    ['Relative Lipid Content / -', 'Ultimate Collagen Stress / MPa'],
    ['Ultimate Stress / MPa', 'Ultimate Strain / -'],
    ['Ultimate Stress / MPa', 'Apparent Modulus Demineralized / MPa'],
    ['Ultimate Stress / MPa', 'Modulus Demineralized / MPa'],
    ['Ultimate Stress / MPa', 'Density / g/cm³'],
    ['Ultimate Stress / MPa', 'Mineral weight fraction / -'],
    ['Ultimate Stress / MPa', 'Organic weight fraction / -'],
    ['Ultimate Stress / MPa', 'Water weight fraction / -'],
    ['Ultimate Stress / MPa', 'Bone Volume Fraction / -'],
    ['Ultimate Stress / MPa', 'Bone Mineral Density / mg HA / cm³'],
    ['Ultimate Stress / MPa', 'Tissue Mineral Density / mg HA / cm³'],
    ['Ultimate Stress / MPa', 'Mineral to Matrix Ratio v2/a3 / -'],
    ['Ultimate Stress / MPa', 'Mineral to Matrix Ratio v1/a1 / -'],
    ['Ultimate Stress / MPa', 'Crystallinity / -'],
    ['Ultimate Stress / MPa', 'Collagen dis/order / -'],
    ['Ultimate Stress / MPa', 'Matrix maturity / -'],
    ['Ultimate Stress / MPa', 'Relative Pyridinoline Content / -'],
    ['Ultimate Stress / MPa', 'Relative Proteoglycan Content / -'],
    ['Ultimate Stress / MPa', 'Relative Lipid Content / -'],
    ['Ultimate Strain / -', 'Apparent Modulus Demineralized / MPa'],
    ['Ultimate Strain / -', 'Modulus Demineralized / MPa'],
    ['Ultimate Strain / -', 'Density / g/cm³'],
    ['Ultimate Strain / -', 'Mineral weight fraction / -'],
    ['Ultimate Strain / -', 'Organic weight fraction / -'],
    ['Ultimate Strain / -', 'Water weight fraction / -'],
    ['Ultimate Strain / -', 'Bone Volume Fraction / -'],
    ['Ultimate Strain / -', 'Bone Mineral Density / mg HA / cm³'],
    ['Ultimate Strain / -', 'Tissue Mineral Density / mg HA / cm³'],
    ['Ultimate Strain / -', 'Mineral to Matrix Ratio v2/a3 / -'],
    ['Ultimate Strain / -', 'Mineral to Matrix Ratio v1/a1 / -'],
    ['Ultimate Strain / -', 'Crystallinity / -'],
    ['Ultimate Strain / -', 'Collagen dis/order / -'],
    ['Ultimate Strain / -', 'Matrix maturity / -'],
    ['Ultimate Strain / -', 'Relative Pyridinoline Content / -'],
    ['Ultimate Strain / -', 'Relative Proteoglycan Content / -'],
    ['Ultimate Strain / -', 'Relative Lipid Content / -'],
    ['Apparent Modulus Demineralized / MPa', 'Modulus Demineralized / MPa'],
    ['Apparent Modulus Demineralized / MPa', 'Density / g/cm³'],
    ['Apparent Modulus Demineralized / MPa', 'Mineral weight fraction / -'],
    ['Apparent Modulus Demineralized / MPa', 'Organic weight fraction / -'],
    ['Apparent Modulus Demineralized / MPa', 'Water weight fraction / -'],
    ['Apparent Modulus Demineralized / MPa', 'Bone Volume Fraction / -'],
    ['Apparent Modulus Demineralized / MPa', 'Bone Mineral Density / mg HA / cm³'],
    ['Apparent Modulus Demineralized / MPa', 'Tissue Mineral Density / mg HA / cm³'],
    ['Apparent Modulus Demineralized / MPa', 'Mineral to Matrix Ratio v2/a3 / -'],
    ['Apparent Modulus Demineralized / MPa', 'Mineral to Matrix Ratio v1/a1 / -'],
    ['Apparent Modulus Demineralized / MPa', 'Crystallinity / -'],
    ['Apparent Modulus Demineralized / MPa', 'Collagen dis/order / -'],
    ['Apparent Modulus Demineralized / MPa', 'Matrix maturity / -'],
    ['Apparent Modulus Demineralized / MPa', 'Relative Pyridinoline Content / -'],
    ['Apparent Modulus Demineralized / MPa', 'Relative Proteoglycan Content / -'],
    ['Apparent Modulus Demineralized / MPa', 'Relative Lipid Content / -'],
    ['Density / g/cm³', 'Modulus Demineralized / MPa'],
    ['Mineral weight fraction / -', 'Modulus Demineralized / MPa'],
    ['Organic weight fraction / -', 'Modulus Demineralized / MPa'],
    ['Water weight fraction / -', 'Modulus Demineralized / MPa'],
    ['Bone Volume Fraction / -', 'Modulus Demineralized / MPa'],
    ['Bone Mineral Density / mg HA / cm³', 'Modulus Demineralized / MPa'],
    ['Tissue Mineral Density / mg HA / cm³', 'Modulus Demineralized / MPa'],
    ['Mineral to Matrix Ratio v2/a3 / -', 'Modulus Demineralized / MPa'],
    ['Mineral to Matrix Ratio v1/a1 / -', 'Modulus Demineralized / MPa'],
    ['Crystallinity / -', 'Modulus Demineralized / MPa'],
    ['Collagen dis/order / -', 'Modulus Demineralized / MPa'],
    ['Matrix maturity / -', 'Modulus Demineralized / MPa'],
    ['Relative Pyridinoline Content / -', 'Modulus Demineralized / MPa'],
    ['Relative Proteoglycan Content / -', 'Modulus Demineralized / MPa'],
    ['Relative Lipid Content / -', 'Modulus Demineralized / MPa'],
    ['Density / g/cm³', 'Mineral weight fraction / -'],
    ['Density / g/cm³', 'Organic weight fraction / -'],
    ['Density / g/cm³', 'Water weight fraction / -'],
    ['Density / g/cm³', 'Bone Volume Fraction / -'],
    ['Density / g/cm³', 'Bone Mineral Density / mg HA / cm³'],
    ['Density / g/cm³', 'Tissue Mineral Density / mg HA / cm³'],
    ['Density / g/cm³', 'Mineral to Matrix Ratio v2/a3 / -'],
    ['Density / g/cm³', 'Mineral to Matrix Ratio v1/a1 / -'],
    ['Density / g/cm³', 'Crystallinity / -'],
    ['Density / g/cm³', 'Collagen dis/order / -'],
    ['Density / g/cm³', 'Matrix maturity / -'],
    ['Density / g/cm³', 'Relative Pyridinoline Content / -'],
    ['Density / g/cm³', 'Relative Proteoglycan Content / -'],
    ['Density / g/cm³', 'Relative Lipid Content / -'],
    ['Mineral weight fraction / -', 'Organic weight fraction / -'],
    ['Mineral weight fraction / -', 'Water weight fraction / -'],
    ['Mineral weight fraction / -', 'Bone Volume Fraction / -'],
    ['Mineral weight fraction / -', 'Bone Mineral Density / mg HA / cm³'],
    ['Mineral weight fraction / -', 'Tissue Mineral Density / mg HA / cm³'],
    ['Mineral weight fraction / -', 'Mineral to Matrix Ratio v2/a3 / -'],
    ['Mineral weight fraction / -', 'Mineral to Matrix Ratio v1/a1 / -'],
    ['Mineral weight fraction / -', 'Crystallinity / -'],
    ['Mineral weight fraction / -', 'Collagen dis/order / -'],
    ['Mineral weight fraction / -', 'Matrix maturity / -'],
    ['Mineral weight fraction / -', 'Relative Pyridinoline Content / -'],
    ['Mineral weight fraction / -', 'Relative Proteoglycan Content / -'],
    ['Mineral weight fraction / -', 'Relative Lipid Content / -'],
    ['Organic weight fraction / -', 'Water weight fraction / -'],
    ['Organic weight fraction / -', 'Bone Volume Fraction / -'],
    ['Organic weight fraction / -', 'Bone Mineral Density / mg HA / cm³'],
    ['Organic weight fraction / -', 'Tissue Mineral Density / mg HA / cm³'],
    ['Organic weight fraction / -', 'Mineral to Matrix Ratio v2/a3 / -'],
    ['Organic weight fraction / -', 'Mineral to Matrix Ratio v1/a1 / -'],
    ['Organic weight fraction / -', 'Crystallinity / -'],
    ['Organic weight fraction / -', 'Collagen dis/order / -'],
    ['Organic weight fraction / -', 'Matrix maturity / -'],
    ['Organic weight fraction / -', 'Relative Pyridinoline Content / -'],
    ['Organic weight fraction / -', 'Relative Proteoglycan Content / -'],
    ['Organic weight fraction / -', 'Relative Lipid Content / -'],
    ['Water weight fraction / -', 'Bone Volume Fraction / -'],
    ['Water weight fraction / -', 'Bone Mineral Density / mg HA / cm³'],
    ['Water weight fraction / -', 'Tissue Mineral Density / mg HA / cm³'],
    ['Water weight fraction / -', 'Mineral to Matrix Ratio v2/a3 / -'],
    ['Water weight fraction / -', 'Mineral to Matrix Ratio v1/a1 / -'],
    ['Water weight fraction / -', 'Crystallinity / -'],
    ['Water weight fraction / -', 'Collagen dis/order / -'],
    ['Water weight fraction / -', 'Matrix maturity / -'],
    ['Water weight fraction / -', 'Relative Pyridinoline Content / -'],
    ['Water weight fraction / -', 'Relative Proteoglycan Content / -'],
    ['Water weight fraction / -', 'Relative Lipid Content / -'],
    ['Bone Volume Fraction / -', 'Bone Mineral Density / mg HA / cm³'],
    ['Bone Volume Fraction / -', 'Tissue Mineral Density / mg HA / cm³'],
    ['Bone Volume Fraction / -', 'Mineral to Matrix Ratio v2/a3 / -'],
    ['Bone Volume Fraction / -', 'Mineral to Matrix Ratio v1/a1 / -'],
    ['Bone Volume Fraction / -', 'Crystallinity / -'],
    ['Bone Volume Fraction / -', 'Collagen dis/order / -'],
    ['Bone Volume Fraction / -', 'Matrix maturity / -'],
    ['Bone Volume Fraction / -', 'Relative Pyridinoline Content / -'],
    ['Bone Volume Fraction / -', 'Relative Proteoglycan Content / -'],
    ['Bone Volume Fraction / -', 'Relative Lipid Content / -'],
    ['Bone Mineral Density / mg HA / cm³', 'Tissue Mineral Density / mg HA / cm³'],
    ['Bone Mineral Density / mg HA / cm³', 'Mineral to Matrix Ratio v2/a3 / -'],
    ['Bone Mineral Density / mg HA / cm³', 'Mineral to Matrix Ratio v1/a1 / -'],
    ['Bone Mineral Density / mg HA / cm³', 'Crystallinity / -'],
    ['Bone Mineral Density / mg HA / cm³', 'Collagen dis/order / -'],
    ['Bone Mineral Density / mg HA / cm³', 'Matrix maturity / -'],
    ['Bone Mineral Density / mg HA / cm³', 'Relative Pyridinoline Content / -'],
    ['Bone Mineral Density / mg HA / cm³', 'Relative Proteoglycan Content / -'],
    ['Bone Mineral Density / mg HA / cm³', 'Relative Lipid Content / -'],
    ['Tissue Mineral Density / mg HA / cm³', 'Mineral to Matrix Ratio v2/a3 / -'],
    ['Tissue Mineral Density / mg HA / cm³', 'Mineral to Matrix Ratio v1/a1 / -'],
    ['Tissue Mineral Density / mg HA / cm³', 'Crystallinity / -'],
    ['Tissue Mineral Density / mg HA / cm³', 'Collagen dis/order / -'],
    ['Tissue Mineral Density / mg HA / cm³', 'Matrix maturity / -'],
    ['Tissue Mineral Density / mg HA / cm³', 'Relative Pyridinoline Content / -'],
    ['Tissue Mineral Density / mg HA / cm³', 'Relative Proteoglycan Content / -'],
    ['Tissue Mineral Density / mg HA / cm³', 'Relative Lipid Content / -'],
    ['Mineral to Matrix Ratio v2/a3 / -', 'Crystallinity / -'],
    ['Mineral to Matrix Ratio v2/a3 / -', 'Collagen dis/order / -'],
    ['Mineral to Matrix Ratio v2/a3 / -', 'Matrix maturity / -'],
    ['Mineral to Matrix Ratio v2/a3 / -', 'Relative Pyridinoline Content / -'],
    ['Mineral to Matrix Ratio v2/a3 / -', 'Relative Proteoglycan Content / -'],
    ['Mineral to Matrix Ratio v2/a3 / -', 'Relative Lipid Content / -'],
    ['Mineral to Matrix Ratio v1/a1 / -', 'Crystallinity / -'],
    ['Mineral to Matrix Ratio v1/a1 / -', 'Mineral to Matrix Ratio v2/a3 / -'],
    ['Mineral to Matrix Ratio v1/a1 / -', 'Collagen dis/order / -'],
    ['Mineral to Matrix Ratio v1/a1 / -', 'Matrix maturity / -'],
    ['Mineral to Matrix Ratio v1/a1 / -', 'Relative Pyridinoline Content / -'],
    ['Mineral to Matrix Ratio v1/a1 / -', 'Relative Proteoglycan Content / -'],
    ['Mineral to Matrix Ratio v1/a1 / -', 'Relative Lipid Content / -'],
    ['Crystallinity / -', 'Collagen dis/order / -'],
    ['Crystallinity / -', 'Matrix maturity / -'],
    ['Crystallinity / -', 'Relative Pyridinoline Content / -'],
    ['Crystallinity / -', 'Relative Proteoglycan Content / -'],
    ['Crystallinity / -', 'Relative Lipid Content / -'],
    ['Collagen dis/order / -', 'Matrix maturity / -'],
    ['Collagen dis/order / -', 'Relative Pyridinoline Content / -'],
    ['Collagen dis/order / -', 'Relative Proteoglycan Content / -'],
    ['Collagen dis/order / -', 'Relative Lipid Content / -'],
    ['Relative Pyridinoline Content / -', 'Relative Proteoglycan Content / -'],
    ['Relative Pyridinoline Content / -', 'Relative Lipid Content / -'],
    ['Relative Proteoglycan Content / -', 'Relative Lipid Content / -']
])

# assign abbreviations to above list of variables
Pair_abbrev1 = list()
Pair_abbrev2 = list()

for i in range(len(Pair)):
    index1 = ColumnNames.loc[ColumnNames['Column Names'] == Pair[0][i]].index[0]
    index2 = ColumnNames.loc[ColumnNames['Column Names'] == Pair[1][i]].index[0]
    Abbrev1 = ColumnNames['Abbreviations'][index1]
    Abbrev2 = ColumnNames['Abbreviations'][index2]
    Pair_abbrev1.append(Abbrev1)
    Pair_abbrev2.append(Abbrev2)

Pair_abbrev1_df = pd.DataFrame(Pair_abbrev1)
Pair_abbrev2_df = pd.DataFrame(Pair_abbrev2)
Pair_abbrev_df = pd.DataFrame()
Pair_abbrev_df['Abbrev_x'] = Pair_abbrev1_df
Pair_abbrev_df['Abbrev_y'] = Pair_abbrev2_df

# loop to iterate through lists of names & create plots
results = list()
j = 0

for i in tqdm(range(len(Pair))):
    x_axis = Pair[0][i]
    y_axis = Pair[1][i]
    x_axis_abbrev = Pair_abbrev_df['Abbrev_x'][i]
    y_axis_abbrev = Pair_abbrev_df['Abbrev_y'][i]
    x_axis_label = AxisLabels.loc[AxisLabels['Abbreviations'] == x_axis_abbrev].iloc[0][0]
    y_axis_label = AxisLabels.loc[AxisLabels['Abbreviations'] == y_axis_abbrev].iloc[0][0]
    if x_axis == 'Age / y':
        Data = df.filter(['Sample ID', x_axis, y_axis, 'Gender'])
    else:
        Data = df.filter(['Sample ID', x_axis, y_axis, 'Gender', 'Age / y'])

    Data = Data[Data[x_axis].notna() & Data[y_axis].notna()]
    Data = Data.reset_index(drop=True)

    Data2Fit = Data.copy()
    Data2Fit.rename(columns={'Sample ID': 'SID', x_axis: x_axis_abbrev, y_axis: y_axis_abbrev}, inplace=True)
    Data2Fit = Data2Fit.set_index('SID')
    FitResults = smf.ols(y_axis_abbrev + ' ~ 1 + ' + x_axis_abbrev, data=Data2Fit).fit()

    # # Manually check correlations
    # x_axis = 'Mineral weight fraction / -'
    # y_axis = 'Modulus Demineralized / MPa'
    # x_axis_abbrev = 'MWF'
    # y_axis_abbrev = 'EC'
    # Data = df.filter(['Sample ID', x_axis, y_axis, 'Gender', 'Age']).dropna()
    # Data2Fit = Data.copy()
    # Data2Fit.rename(columns={'Sample ID': 'SID', x_axis: x_axis_abbrev, y_axis: y_axis_abbrev}, inplace=True)
    # FitResults = smf.ols(y_axis_abbrev + ' ~ 1 + ' + x_axis_abbrev, data=Data2Fit).fit()
    # print(FitResults.summary())
    # testlin = linregress(Data2Fit['USTRE'], Data2Fit['USTRA'])
    # print(testlin)

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
    X = np.matrix(FitResults.model.exog)
    X_Obs = np.sort(np.array(X[:, 1]).reshape(len(X)))
    C = np.matrix(FitResults.cov_params())
    B_0 = np.sqrt(np.diag(np.abs(X * C * X.T)))
    Alpha = 0.95
    t_Alpha = t.interval(Alpha, N - X.shape[1] - 1)
    CI_Line_u = Y_Fit + t_Alpha[0] * SE * B_0
    CI_Line_o = Y_Fit + t_Alpha[1] * SE * B_0
    Sorted_CI_u = CI_Line_u[np.argsort(FitResults.model.exog[:, 1])]
    Sorted_CI_o = CI_Line_o[np.argsort(FitResults.model.exog[:, 1])]
    Y_Fit_df = pd.DataFrame(Y_Fit)
    Y_Obs_df = pd.DataFrame(Y_Obs)
    RMSE = rmse(Y_Obs_df, Y_Fit_df)
    cv = round(100 * (RMSE / statistics.mean(Y_Obs))[0], 2)

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=300, sharey=True, sharex=True)
    male_age = Data[Data['Gender'] == 'M']['Age / y']
    female_age = Data[Data['Gender'] == 'F']['Age / y']
    X_np = np.array(X)
    Y_Obs_np = np.array(Y_Obs)

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
        p_plot = '$p < 0.001$'
    elif p == 0.001:
        p_plot = '$p = 0.001$'
    elif p < 0.01:
        p_plot = '$p < 0.01$'
    elif p == 0.01:
        p_plot = '$p = 0.01$'
    elif p < 0.05:
        p_plot = '$p < 0.05$'
    elif p == 0.05:
        p_plot = '$p = 0.05$'
    else:
        p_plot = '$p > 0.05$'

    if p < 0.001:
        p = '{:.1e}'.format(p)
    else:
        p = round(p, 3)

    if float(R2) < 0.01:
        R2 = float(R2)
        R2 = '{:.1e}'.format(R2)
    else:
        R2 = float(R2)
        R2 = round(R2, 2)

    # Positions of annotations
    YposCI = 0.025
    YposCV = YposCI + 0.075
    YposN = YposCV + 0.075

    # y-axis limitation values used for plotting
    ylim_min = Y_Obs.min() - (Y_Obs.max() - Y_Obs.min()) * 0.5
    ylim_max = Y_Obs.max() + (Y_Obs.max() - Y_Obs.min()) * 0.1

    # if p-value < 0.05 create fit curve; use colormap if age is not on x-axis
    if float(p) <= 0.05:
        if x_axis != 'Age / y':
            Axes.plot(X[:, 1], Y_Fit, color=(1, 0, 0), linewidth=1)
            Axes.scatter(X_np[:, 1][Data['Gender'] == 'F'], Y_Obs_np[Data['Gender'] == 'F'],
                         c=list(tuple(female_age.tolist())), cmap='plasma_r', vmin=Data['Age / y'].min(),
                         vmax=Data['Age / y'].max(), label='F', marker='o', s=50)
            Axes.scatter(X_np[:, 1][Data['Gender'] == 'M'], Y_Obs_np[Data['Gender'] == 'M'],
                         c=list(tuple(male_age.tolist())), cmap='plasma_r', vmin=Data['Age / y'].min(),
                         vmax=Data['Age / y'].max(), label='M', marker='s', s=50)
            Axes.scatter(X_np[:, 1][Data['Gender'].isnull()], Y_Obs_np[Data['Gender'].isnull()], color=(0, 0, 0),
                         label='N/A', marker='^', s=50)
            Axes.plot([], ' ', label=r'$N$ = ' + str(N) + ', 'r'$R^2$ = ' + str(R2))
            Axes.plot([], ' ', label=r'$CV$ = ' + str(cv) + ', ' + p_plot)
            Axes.plot([], ' ', label='95% CI [' + str(CI_l) + r'$,$ ' + str(CI_r) + ']')
            Regression_line = FitResults.params[1] * X_np[:, 1] + FitResults.params[0]
            ax = plt.gca()
            PCM = ax.get_children()[2]
            plt.colorbar(PCM, ax=ax, label='Age / y')

            Axes.set_ylabel(y_axis_label)
            Axes.set_xlabel(x_axis_label)

            # scaling
            plt.ylim(ymin=ylim_min, ymax=ylim_max)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
            # ax.yaxis.set_major_locator(ticker.LinearLocator(6))
            plt.subplots_adjust(left=0.17, bottom=0.15, right=0.95, top=0.95)
            plt.legend(loc='lower center',
                       bbox_to_anchor=(0.5, 0.),
                       ncol=2,
                       columnspacing=0.1,
                       handletextpad=0.1,
                       handlelength=1,
                       labelspacing=0.3)
            plt.rcParams['figure.figsize'] = (5.5, 4.0)
            plt.rcParams.update({'font.size': 12})
            plt.savefig(os.path.join(Savepath, Data2Fit.columns[0] + '_' + Data2Fit.columns[1] + '.jpg'),
                        dpi=1200, format='jpg')
            # plt.show()
            plt.close()

        # if p < 0.05 and age is shown on x-axis, don't use colormap
        else:
            sns.regplot(x=FitResults.model.exog[:, 1], y=Y_Obs, ax=Axes, scatter=False, color=(0, 1, 0),
                        line_kws={'color': 'red', 'linewidth': 1}, )  # set background color of confidence interval here
            Axes.plot(X[:, 1][Data['Gender'] == 'F'], Y_Obs[Data['Gender'] == 'F'], linestyle='none', marker='o',
                      color=(0, 0, 0), fillstyle='none', label='F', markersize=9)
            Axes.plot(X[:, 1][Data['Gender'] == 'M'], Y_Obs[Data['Gender'] == 'M'], linestyle='none', marker='x',
                      color=(0, 0, 0), fillstyle='none', label='M', markersize=9)
            Axes.plot([], ' ', label=' ')
            Axes.plot([], ' ', label=r'$N$ = ' + str(N) + ', 'r'$R^2$ = ' + str(R2))
            Axes.plot([], ' ', label=r'$CV$ = ' + str(cv) + ', ' + p_plot)
            Axes.plot([], ' ', label='95% CI [' + str(CI_l) + r'$,$ ' + str(CI_r) + ']')

            Axes.set_ylabel(y_axis_label)
            Axes.set_xlabel(x_axis_label)

            plt.xlim(xmin=55, xmax=95)
            plt.ylim(ymin=ylim_min, ymax=ylim_max)
            Axes.yaxis.set_major_locator(ticker.MaxNLocator(6))

            plt.subplots_adjust(left=0.17, bottom=0.15, right=0.95, top=0.95)
            plt.legend(loc='lower center',
                       bbox_to_anchor=(0.5, 0.),
                       ncol=2,
                       columnspacing=0.1,
                       handletextpad=0.1,
                       handlelength=1,
                       labelspacing=0.3)
            plt.rcParams['figure.figsize'] = (5.5, 4.0)
            plt.rcParams.update({'font.size': 12})
            plt.savefig(os.path.join(Savepath, Data2Fit.columns[0] + '_' + Data2Fit.columns[1] + '.jpg'),
                        dpi=1200, format='jpg', pad_inches=0)
            # plt.show()
            plt.close()
            j = j + 1

    # if p-value > 0.05, no fit will be drawn; if age is not shown on x-axis colormap will be used
    else:
        if x_axis != 'Age / y':
            Axes.scatter(X_np[:, 1][Data['Gender'] == 'F'], Y_Obs_np[Data['Gender'] == 'F'],
                         c=list(tuple(female_age.tolist())), cmap='plasma_r', vmin=Data['Age / y'].min(),
                         vmax=Data['Age / y'].max(), label='F', marker='o', s=50)
            Axes.scatter(X_np[:, 1][Data['Gender'] == 'M'], Y_Obs_np[Data['Gender'] == 'M'],
                         c=list(tuple(male_age.tolist())), cmap='plasma_r', vmin=Data['Age / y'].min(),
                         vmax=Data['Age / y'].max(), label='M', marker='s', s=50)
            Axes.scatter(X_np[:, 1][Data['Gender'].isnull()], Y_Obs_np[Data['Gender'].isnull()], color=(0, 0, 0),
                         label='N/A', marker='^', s=50)
            Axes.plot([], ' ', label=r'$N$ = ' + str(N) + ', 'r'$R^2$ = ' + str(R2))
            Axes.plot([], ' ', label=r'$CV$ = ' + str(cv) + ', ' + p_plot)
            Axes.plot([], ' ', label='95% CI [' + str(CI_l) + r'$,$ ' + str(CI_r) + ']')
            ax = plt.gca()
            PCM = ax.get_children()[0]
            plt.colorbar(PCM, ax=ax, label='Age / y')

            Axes.set_ylabel(y_axis_label)
            Axes.set_xlabel(x_axis_label)

            plt.ylim(ymin=ylim_min, ymax=ylim_max)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
            plt.subplots_adjust(left=0.17, bottom=0.15, right=0.95, top=0.95)
            plt.legend(loc='lower center',
                       bbox_to_anchor=(0.5, 0.),
                       ncol=2,
                       columnspacing=0.1,
                       handletextpad=0.1,
                       handlelength=1,
                       labelspacing=0.3)
            plt.rcParams['figure.figsize'] = (5.5, 4.0)
            plt.rcParams.update({'font.size': 12})
            plt.savefig(os.path.join(Savepath, Data2Fit.columns[0] + '_' + Data2Fit.columns[1] + '.jpg'),
                        dpi=1200, format='jpg', pad_inches=0)
            # plt.show()
            plt.close()
            j = j + 1

        # if p > 0.05 and age is shown on x-axis, don't use colormap
        else:
            Axes.plot(X[:, 1][Data['Gender'] == 'F'], Y_Obs[Data['Gender'] == 'F'], linestyle='none', marker='o',
                      color=(0, 0, 0), fillstyle='none', label='F', markersize=10)
            Axes.plot(X[:, 1][Data['Gender'] == 'M'], Y_Obs[Data['Gender'] == 'M'], linestyle='none', marker='x',
                      color=(0, 0, 0), fillstyle='none', label='M', markersize=10)
            Axes.plot([], ' ', label=' ')
            Axes.plot([], ' ', label=r'$N$ = ' + str(N) + ', 'r'$R^2$ = ' + str(R2))
            Axes.plot([], ' ', label=r'$CV$ = ' + str(cv) + ', ' + p_plot)
            Axes.plot([], ' ', label='95% CI [' + str(CI_l) + r'$,$ ' + str(CI_r) + ']')

            Axes.set_ylabel(y_axis_label)
            Axes.set_xlabel(x_axis_label)

            plt.xlim(xmin=55, xmax=95)
            plt.ylim(ymin=Y_Obs.min() - (Y_Obs.max() - Y_Obs.min()) * 0.6, ymax=ylim_max)
            Axes.yaxis.set_major_locator(ticker.MaxNLocator(6))
            plt.subplots_adjust(left=0.17, bottom=0.15, right=0.95, top=0.95)
            plt.legend(loc='lower center',
                       bbox_to_anchor=(0.5, 0.),
                       ncol=2,
                       columnspacing=0.1,
                       handletextpad=0.1,
                       handlelength=1,
                       labelspacing=0.3)
            plt.rcParams['figure.figsize'] = (5.5, 4.0)
            plt.rcParams.update({'font.size': 12})
            plt.savefig(os.path.join(Savepath, Data2Fit.columns[0] + '_' + Data2Fit.columns[1] + '.jpg'),
                        dpi=1200, format='jpg', pad_inches=0)
            # plt.show()
            plt.close()
            j = j + 1

    # Put everything into growing list and convert to DataFrame that is saved as .csv file
    values = [x_axis, y_axis, p, SE, R2, N, CI_l, CI_r, SE, RMSE]
    results.append(values)

result_dir = pd.DataFrame(results, columns=['X-axis', 'Y-axis', 'p-value', '\u03C3\u2091\u209B\u209C', 'R\u00B2', 'N',
                                            'lower bound 95% CI', 'upper bound 95% CI', 'Standard error',
                                            'Root mean square error'])
result_dir.to_csv(Results_path + '/04_Plots/ResultsPlots.csv', index=False)
