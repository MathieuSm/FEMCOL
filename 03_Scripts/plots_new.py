# This script plots various variables against each other, Data is retrieved from ResultsOverview.csv file

# Import standard packages
import numpy as np  # Used to do arrays (matrices) computations namely
import pandas as pd  # Used to manage data frames
import matplotlib.pyplot as plt  # Used to perform plots
import statsmodels.formula.api as smf  # Used for statistical analysis (ols here)
import os  # Used to manage path variables
from scipy.stats.distributions import t  # Used to compute confidence intervals
import seaborn as sns  # Used to create regression lines with confidence bands
import statistics  # Used to calculate statistical measures
from statsmodels.tools.eval_measures import rmse  # Used to evaluate rmse
from tqdm import tqdm  # Used to track script progression while running
import matplotlib.ticker as ticker

# Set directory & load data
Cwd = os.getcwd()
Results_path = str(os.path.dirname(Cwd) + '/04_Results')
results_overview = str(os.path.dirname(Cwd) + '/04_Results/ResultsOverview.csv')
Savepath = Results_path + '/04_Plots/Individual'

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
     'Apparent Modulus Demineralized / MPa': 'Apparent Modulus Demineralized E$_{c}^{app}$ / MPa',
     'Modulus Demineralized / MPa': 'Modulus Demineralized E$_{c}$ / MPa',
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
                       'USTRA', 'EAPPC', 'EC', 'D', 'OW', 'MW', 'WW', 'WFM', 'WFO', 'WFW', 'BVTV', 'BMD', 'TMD', 'BMC',
                       'ABMean', 'ABMin', 'MMRv2a3', 'MMRv1a1', 'CRY', 'COLDIS', 'MATMAT', 'RPyC', 'RProC', 'RLC',
                       'EAPPFE', 'YSTREFE', 'USTREFE', 'HCMean', 'HCStd', 'OCMean', 'OCStd', 'CLMean', 'CLStd']
ColumnNames['Abbreviations'] = column_names_abbrev
AxisLabels['Abbreviations'] = column_names_abbrev

# Pair has the Format [x-axis, y-axis]; stress & moduli need to be on y-axis for non-age plots
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
    FitResults_f = smf.ols(y_axis_abbrev + ' ~ 1 + ' + x_axis_abbrev, data=Data2Fit[Data2Fit['Gender'] == 'F']).fit()
    FitResults_m = smf.ols(y_axis_abbrev + ' ~ 1 + ' + x_axis_abbrev, data=Data2Fit[Data2Fit['Gender'] == 'M']).fit()
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

    # Calculate R^2, p-value, 95% CI, SE, N for female dataset
    Y_Obs_f = FitResults_f.model.endog
    Y_Fit_f = FitResults_f.fittedvalues

    E_f = Y_Obs_f - Y_Fit_f
    RSS_f = np.sum(E_f ** 2)
    SE_f = np.sqrt(RSS_f / FitResults_f.df_resid)
    CI_l_f = FitResults_f.conf_int()[0][1]
    CI_r_f = FitResults_f.conf_int()[1][1]
    N_f = int(FitResults_f.nobs)
    R2_f = FitResults_f.rsquared
    p_f = FitResults_f.pvalues[1]
    X_f = np.matrix(FitResults_f.model.exog)
    X_Obs_f = np.sort(np.array(X_f[:, 1]).reshape(len(X_f)))
    C_f = np.matrix(FitResults_f.cov_params())
    B_0_f = np.sqrt(np.diag(np.abs(X_f * C_f * X_f.T)))
    Alpha = 0.95
    t_Alpha_f = t.interval(Alpha, N_f - X_f.shape[1] - 1)
    CI_Line_u_f = Y_Fit_f + t_Alpha_f[0] * SE_f * B_0_f
    CI_Line_o_f = Y_Fit_f + t_Alpha_f[1] * SE_f * B_0_f
    Sorted_CI_u_f = CI_Line_u_f[np.argsort(FitResults_f.model.exog[:, 1])]
    Sorted_CI_o_f = CI_Line_o_f[np.argsort(FitResults_f.model.exog[:, 1])]
    Y_Fit_df_f = pd.DataFrame(Y_Fit_f)
    Y_Obs_df_f = pd.DataFrame(Y_Obs_f)
    RMSE_f = rmse(Y_Obs_df_f, Y_Fit_df_f)
    cv_f = round(100 * (RMSE_f / statistics.mean(Y_Obs_f))[0], 2)
    X_np_f = np.array(X_f)
    Y_Obs_np_f = np.array(Y_Obs_f)

    # Calculate R^2, p-value, 95% CI, SE, N for male dataset
    Y_Obs_m = FitResults_m.model.endog
    Y_Fit_m = FitResults_m.fittedvalues

    E_m = Y_Obs_m - Y_Fit_m
    RSS_m = np.sum(E_m ** 2)
    SE_m = np.sqrt(RSS_m / FitResults_m.df_resid)
    CI_l_m = FitResults_m.conf_int()[0][1]
    CI_r_m = FitResults_m.conf_int()[1][1]
    N_m = int(FitResults_m.nobs)
    R2_m = FitResults_m.rsquared
    p_m = FitResults_m.pvalues[1]
    X_m = np.matrix(FitResults_m.model.exog)
    X_Obs_m = np.sort(np.array(X_m[:, 1]).reshape(len(X_m)))
    C_m = np.matrix(FitResults_m.cov_params())
    B_0_m = np.sqrt(np.diag(np.abs(X_m * C_m * X_m.T)))
    t_Alpha_m = t.interval(Alpha, N_m - X_m.shape[1] - 1)
    CI_Line_u_m = Y_Fit_m + t_Alpha_m[0] * SE_m * B_0_m
    CI_Line_o_m = Y_Fit_m + t_Alpha_m[1] * SE_m * B_0_m
    Sorted_CI_u_m = CI_Line_u_m[np.argsort(FitResults_m.model.exog[:, 1])]
    Sorted_CI_o_m = CI_Line_o_m[np.argsort(FitResults_m.model.exog[:, 1])]
    Y_Fit_df_m = pd.DataFrame(Y_Fit_m)
    Y_Obs_df_m = pd.DataFrame(Y_Obs_m)
    RMSE_m = rmse(Y_Obs_df_m, Y_Fit_df_m)
    cv_m = round(100 * (RMSE_m / statistics.mean(Y_Obs_m))[0], 2)
    X_np_m = np.array(X_m)
    Y_Obs_np_m = np.array(Y_Obs_m)

    X = np.matrix(FitResults.model.exog)
    X_np = np.array(X)
    Y_Obs = FitResults.model.endog
    Y_Obs_np = np.array(Y_Obs)

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=300, sharey=True, sharex=True)
    male_age = Data[Data['Gender'] == 'M']['Age / y']
    female_age = Data[Data['Gender'] == 'F']['Age / y']

    # Requirements for rounding of female dataset: depending on value
    if abs(CI_l_f) >= 100:
        CI_l_f = round(CI_l_f)
    elif abs(CI_l_f) >= 1:
        CI_l_f = round(CI_l_f, 1)
    elif abs(CI_l_f) == 0:
        CI_l_f = int(CI_l_f)
    elif abs(CI_l_f) >= 0.001:
        CI_l_f = round(CI_l_f, 3)
    elif abs(CI_l_f) < 0.001:
        CI_l_f = '{:.2e}'.format(CI_l_f)

    if abs(CI_r_f) >= 100:
        CI_r_f = round(CI_r_f)
    elif abs(CI_r_f) >= 1:
        CI_r_f = round(CI_r_f, 1)
    elif abs(CI_r_f) == 0:
        CI_r_f = int(CI_r_f)
    elif abs(CI_r_f) >= 0.001:
        CI_r_f = round(CI_r_f, 3)
    elif abs(CI_r_f) < 0.001:
        CI_r_f = '{:.2e}'.format(CI_r_f)

    if abs(SE_f) >= 100:
        SE_f = round(SE_f)
    elif abs(SE_f) >= 1:
        SE_f = round(SE_f, 1)
    elif abs(SE_f) == 0:
        SE_f = int(SE_f)
    elif abs(SE_f) >= 0.001:
        SE_f = round(SE_f, 3)
    elif abs(SE_f) < 0.001:
        SE_f = '{:.1e}'.format(SE_f)

    if p_f < 0.001:
        p_plot_f = '$p < 0.001$'
    elif p_f == 0.001:
        p_plot_f = '$p = 0.001$'
    elif p_f < 0.01:
        p_plot_f = '$p < 0.01$'
    elif p_f == 0.01:
        p_plot_f = '$p = 0.01$'
    elif p_f < 0.05:
        p_plot_f = '$p < 0.05$'
    elif p_f == 0.05:
        p_plot_f = '$p = 0.05$'
    else:
        p_plot_f = '$p > 0.05$'

    if p_f < 0.001:
        p_f = '{:.1e}'.format(p_f)
    else:
        p_f = round(p_f, 3)

    if float(R2_f) < 0.01:
        R2_f = float(R2_f)
        R2_f = '{:.1e}'.format(R2_f)
    else:
        R2_f = float(R2_f)
        R2_f = round(R2_f, 2)

    # Requirements for rounding of male dataset: depending on value
    if abs(CI_l_m) >= 100:
        CI_l_m = round(CI_l_m)
    elif abs(CI_l_m) >= 1:
        CI_l_m = round(CI_l_m, 1)
    elif abs(CI_l_m) == 0:
        CI_l_m = int(CI_l_m)
    elif abs(CI_l_m) >= 0.001:
        CI_l_m = round(CI_l_m, 3)
    elif abs(CI_l_m) < 0.001:
        CI_l_m = '{:.2e}'.format(CI_l_m)

    if abs(CI_r_m) >= 100:
        CI_r_m = round(CI_r_m)
    elif abs(CI_r_m) >= 1:
        CI_r_m = round(CI_r_m, 1)
    elif abs(CI_r_m) == 0:
        CI_r_m = int(CI_r_m)
    elif abs(CI_r_m) >= 0.001:
        CI_r_m = round(CI_r_m, 3)
    elif abs(CI_r_m) < 0.001:
        CI_r_m = '{:.2e}'.format(CI_r_m)

    if abs(SE_m) >= 100:
        SE_m = round(SE_m)
    elif abs(SE_m) >= 1:
        SE_m = round(SE_m, 1)
    elif abs(SE_m) == 0:
        SE_m = int(SE_m)
    elif abs(SE_m) >= 0.001:
        SE_m = round(SE_m, 3)
    elif abs(SE_m) < 0.001:
        SE_m = '{:.1e}'.format(SE_m)

    if p_m < 0.001:
        p_plot_m = '$p < 0.001$'
    elif p_m == 0.001:
        p_plot_m = '$p = 0.001$'
    elif p_m < 0.01:
        p_plot_m = '$p < 0.01$'
    elif p_m == 0.01:
        p_plot_m = '$p = 0.01$'
    elif p_m < 0.05:
        p_plot_m = '$p < 0.05$'
    elif p_m == 0.05:
        p_plot_m = '$p = 0.05$'
    else:
        p_plot_m = '$p > 0.05$'

    if p_m < 0.001:
        p_m = '{:.1e}'.format(p_m)
    else:
        p_m = round(p_m, 3)

    if float(R2_m) < 0.01:
        R2_m = float(R2_m)
        R2_m = '{:.1e}'.format(R2_m)
    else:
        R2_m = float(R2_m)
        R2_m = round(R2_m, 2)

    # Positions of annotations
    YposCI = 0.025
    YposCV = YposCI + 0.075
    YposN = YposCV + 0.075

    # y-axis limitation values used for plotting
    if Y_Obs_f.min() < Y_Obs_m.min():
        Y_Obs_min = Y_Obs_f.min()
    else:
        Y_Obs_min = Y_Obs_m.min()
    if Y_Obs_f.max() > Y_Obs_m.max():
        Y_Obs_max = Y_Obs_f.max()
    else:
        Y_Obs_max = Y_Obs_m.max()

    ylim_min = Y_Obs_min - (Y_Obs_max - Y_Obs_min) * 0.5
    ylim_max = Y_Obs_max + (Y_Obs_max - Y_Obs_min) * 0.1

    # if p-value smaller than 0.05 for both male and female dataset, create fit curve; if variable 'Age' should not
    # be plotted on main axis, no colormap will be used
    if ((float(p_f) <= 0.05) & (float(p_m) <= 0.05)):
        if x_axis != 'Age / y':
            Axes.plot(X_f[:, 1], Y_Fit_f, color=(1, 0, 0), linewidth=1, linestyle='solid')
            Axes.plot(X_m[:, 1], Y_Fit_m, color=(0, 0, 1), linewidth=1, linestyle='solid')
            Axes.scatter(X_np_f[:, 1], Y_Obs_np_f, c=list(tuple(female_age.tolist())), cmap='plasma_r',
                         vmin=Data['Age / y'].min(), vmax=Data['Age / y'].max(), marker='o', s=50)
            Axes.scatter(X_np_m[:, 1], Y_Obs_np_m, c=list(tuple(male_age.tolist())), cmap='plasma_r',
                         vmin=Data['Age / y'].min(), vmax=Data['Age / y'].max(), marker='s', s=50)
            Axes.plot([], linestyle='solid', markerfacecolor='none', mec='black', color='red', marker='o', label='F')
            Axes.plot([], linestyle='solid', markerfacecolor='none', mec='black', color='blue', marker='s', label='M')
            Axes.scatter(X_np[:, 1][Data['Gender'].isnull()], Y_Obs_np[Data['Gender'].isnull()], color=(0, 0, 0),
                         label='N/A', marker='^', s=50)
            Axes.plot([], ' ', label=r'$R^2$ = ' + str(R2_f) + ', ' + p_plot_f)
            Axes.plot([], ' ', label=r'$R^2$ = ' + str(R2_m) + ', ' + p_plot_m)

            ax = plt.gca()
            PCM = ax.get_children()[2]  # [x]: number of rows until first usage of colormap --> Axes.scatter(.. cmap=)
            plt.colorbar(PCM, ax=ax, label='Age / y')

            Axes.set_ylabel(y_axis_label)
            Axes.set_xlabel(x_axis_label)

            # scaling
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
                        dpi=1200, format='jpg')
            # plt.show()
            plt.close()

        # don't use colormap if age is plotted on x-axis
        else:
            sns.regplot(x=FitResults_f.model.exog[:, 1], y=Y_Obs_f, ax=Axes, scatter=False, color=(0, 1, 0),
                        line_kws={'color': 'red', 'linewidth': 1, 'linestyle': 'solid'})  # set background color of confidence interval here
            sns.regplot(x=FitResults_m.model.exog[:, 1], y=Y_Obs_m, ax=Axes, scatter=False, color=(0, 0, 1),
                        line_kws={'color': 'blue', 'linewidth': 1, 'linestyle': 'solid'})
            Axes.plot(X_f[:, 1], Y_Obs_f, linestyle='none', marker='o', color=(1, 0, 0), fillstyle='none',
                      markersize=9)
            Axes.plot(X_m[:, 1], Y_Obs_m, linestyle='none', marker='x', color=(0, 0, 1), fillstyle='none',
                      markersize=9)
            Axes.plot([], linestyle='solid', markerfacecolor='none', color='red', marker='o', label='F')
            Axes.plot([], linestyle='solid', markerfacecolor='none', color='blue', marker='x', label='M')
            Axes.plot([], ' ', label=r'$R^2$ = ' + str(R2_f) + ', ' + p_plot_f)
            Axes.plot([], ' ', label=r'$R^2$ = ' + str(R2_m) + ', ' + p_plot_m)

            Axes.set_ylabel(y_axis_label)
            Axes.set_xlabel(x_axis_label)

            # scaling
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

    # if p-value smaller than 0.05 for female and larger than 0.05 for male dataset, create fit curve only for female;
    # if variable 'Age' should not be plotted on main axis, no colormap will be used
    if ((float(p_f) <= 0.05) & (float(p_m) > 0.05)):
        if x_axis != 'Age / y':
            Axes.plot(X_f[:, 1], Y_Fit_f, color=(1, 0, 0), linewidth=1, linestyle='solid')
            Axes.scatter(X_np_f[:, 1], Y_Obs_np_f, c=list(tuple(female_age.tolist())), cmap='plasma_r',
                         vmin=Data['Age / y'].min(), vmax=Data['Age / y'].max(), marker='o', s=50)
            Axes.plot([], linestyle='solid', markerfacecolor='none', mec='black', color='red', marker='o', label='F')
            Axes.scatter(X_np_m[:, 1], Y_Obs_np_m, c=list(tuple(male_age.tolist())), cmap='plasma_r',
                         vmin=Data['Age / y'].min(), vmax=Data['Age / y'].max(), marker='s', s=50)
            Axes.plot([], linestyle='none', markerfacecolor='none', mec='black', marker='s', label='M')
            Axes.scatter(X_np[:, 1][Data['Gender'].isnull()], Y_Obs_np[Data['Gender'].isnull()], color=(0, 0, 0),
                         label='N/A', marker='^', s=50)
            Axes.plot([], ' ', label=r'$R^2$ = ' + str(R2_f) + ', ' + p_plot_f)
            Axes.plot([], ' ', label=r'$R^2$ = ' + str(R2_m) + ', ' + p_plot_m)
            ax = plt.gca()
            PCM = ax.get_children()[1]
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

        # don't use colormap if age is plotted on x-axis
        else:
            sns.regplot(x=FitResults_f.model.exog[:, 1], y=Y_Obs_f, ax=Axes, scatter=False, color=(0, 1, 0),
                        line_kws={'color': 'red', 'linewidth': 1,
                                  'linestyle': 'solid'})  # set background color of confidence interval here
            Axes.plot(X_f[:, 1], Y_Obs_f, linestyle='none', marker='o', color=(1, 0, 0), fillstyle='none',
                      markersize=9)
            Axes.plot([], linestyle='solid', markerfacecolor='none', color='red', marker='o', label='F')
            Axes.plot(X_m[:, 1], Y_Obs_m, linestyle='none', marker='x', color=(0, 0, 1), fillstyle='none', label='M',
                      markersize=9)
            Axes.plot([], ' ', label=' ')
            Axes.plot([], ' ', label=r'$R^2$ = ' + str(R2_f) + ', ' + p_plot_f)
            Axes.plot([], ' ', label=r'$R^2$ = ' + str(R2_m) + ', ' + p_plot_m)

            Axes.set_ylabel(y_axis_label)
            Axes.set_xlabel(x_axis_label)

            # scaling
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

    # if p-value smaller than 0.05 for male and larger than 0.05 for female dataset, create fit curve only for male;
    # if variable 'Age' should not be plotted on main axis, no colormap will be used
    if ((float(p_f) > 0.05) & (float(p_m) <= 0.05)):
        if x_axis != 'Age / y':
            Axes.plot(X_m[:, 1], Y_Fit_m, color=(0, 0, 1), linewidth=1, linestyle='solid')
            Axes.scatter(X_np_f[:, 1], Y_Obs_np_f, c=list(tuple(female_age.tolist())), cmap='plasma_r',
                         vmin=Data['Age / y'].min(), vmax=Data['Age / y'].max(), marker='o', s=50)
            Axes.plot([], linestyle='none', markerfacecolor='none', mec='black', marker='o', label='F')
            Axes.scatter(X_np_m[:, 1], Y_Obs_np_m, c=list(tuple(male_age.tolist())), cmap='plasma_r',
                         vmin=Data['Age / y'].min(), vmax=Data['Age / y'].max(), marker='s', s=50)
            Axes.plot([], linestyle='solid', markerfacecolor='none', mec='black', color='blue', marker='s', label='M')
            Axes.scatter(X_np[:, 1][Data['Gender'].isnull()], Y_Obs_np[Data['Gender'].isnull()], color=(0, 0, 0),
                         label='N/A', marker='^', s=50)
            Axes.plot([], ' ', label=r'$R^2$ = ' + str(R2_f) + ', ' + p_plot_f)
            Axes.plot([], ' ', label=r'$R^2$ = ' + str(R2_m) + ', ' + p_plot_m)
            ax = plt.gca()
            PCM = ax.get_children()[1]
            plt.colorbar(PCM, ax=ax, label='Age / y')

            Axes.set_ylabel(y_axis_label)
            Axes.set_xlabel(x_axis_label)

            # scaling
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
                        dpi=1200, format='jpg')
            # plt.show()
            plt.close()

        # don't use colormap if age is plotted on x-axis
        else:
            sns.regplot(x=FitResults_m.model.exog[:, 1], y=Y_Obs_m, ax=Axes, scatter=False, color=(0, 1, 0),
                        line_kws={'color': 'blue', 'linewidth': 1, 'linestyle': 'solid'})
            Axes.plot(X_f[:, 1], Y_Obs_f, linestyle='none', marker='o', color=(1, 0, 0), fillstyle='none', label='F',
                      markersize=9)
            Axes.plot(X_m[:, 1], Y_Obs_m, linestyle='none', marker='x', color=(0, 0, 1), fillstyle='none',
                      markersize=9)
            Axes.plot([], linestyle='solid', markerfacecolor='none', color='blue', marker='o', label='M')
            Axes.plot([], ' ', label=' ')
            Axes.plot([], ' ', label=r'$R^2$ = ' + str(R2_f) + ', ' + p_plot_f)
            Axes.plot([], ' ', label=r'$R^2$ = ' + str(R2_m) + ', ' + p_plot_m)

            Axes.set_ylabel(y_axis_label)
            Axes.set_xlabel(x_axis_label)

            # scaling
            plt.xlim(xmin=55, xmax=95)
            plt.ylim(ymin=ylim_min, ymax=ylim_max)
            Axes.yaxis.set_major_locator(ticker.MaxNLocator(6))
            # Axes.yaxis.set_major_locator(ticker.LinearLocator(6))
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

    # if p-value greater than 0.05 for both male and female datasets, don't create fit curve; if variable 'Age' should
    # not be plotted on main axis, no colormap will be used
    if ((float(p_f) > 0.05) & (float(p_m) > 0.05)):
        if x_axis != 'Age / y':
            Axes.scatter(X_np_f[:, 1], Y_Obs_np_f, c=list(tuple(female_age.tolist())), cmap='plasma_r',
                         vmin=Data['Age / y'].min(), vmax=Data['Age / y'].max(), marker='o', s=50)
            Axes.plot([], linestyle='none', markerfacecolor='none', mec='black', marker='o', label='F')
            Axes.scatter(X_np_m[:, 1], Y_Obs_np_m, c=list(tuple(male_age.tolist())), cmap='plasma_r',
                         vmin=Data['Age / y'].min(), vmax=Data['Age / y'].max(), marker='s', s=50)
            Axes.plot([], linestyle='none', markerfacecolor='none', mec='black', marker='s', label='M')
            Axes.scatter(X_np[:, 1][Data['Gender'].isnull()], Y_Obs_np[Data['Gender'].isnull()], color=(0, 0, 0),
                         label='N/A', marker='^', s=50)
            Axes.plot([], ' ', label=r'$R^2$ = ' + str(R2_f) + ', ' + p_plot_f)
            Axes.plot([], ' ', label=r'$R^2$ = ' + str(R2_m) + ', ' + p_plot_m)
            ax = plt.gca()
            PCM = ax.get_children()[0]
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

        # don't use colormap if age is plotted on x-axis
        else:
            Axes.plot(X_f[:, 1], Y_Obs_f, linestyle='none', marker='o', color=(1, 0, 0), fillstyle='none', label='F',
                      markersize=9)
            Axes.plot(X_m[:, 1], Y_Obs_m, linestyle='none', marker='x', color=(0, 0, 1), fillstyle='none', label='M',
                      markersize=9)
            Axes.plot([], ' ', label=' ')
            Axes.plot([], ' ', label=r'$R^2$ = ' + str(R2_f) + ', ' + p_plot_f)
            Axes.plot([], ' ', label=r'$R^2$ = ' + str(R2_m) + ', ' + p_plot_m)

            Axes.set_ylabel(y_axis_label)
            Axes.set_xlabel(x_axis_label)

            # scaling
            plt.xlim(xmin=55, xmax=95)
            plt.ylim(ymin=ylim_min, ymax=ylim_max)
            Axes.yaxis.set_major_locator(ticker.MaxNLocator(6))
            # Axes.yaxis.set_major_locator(ticker.LinearLocator(6))
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

    # Put everything into growing list and convert to DataFrame that is saved as .csv file
    values = [x_axis, y_axis, p_f, R2_f, N_f, CI_l_f, CI_r_f, SE_f, RMSE_f, p_m, R2_m, N_m, CI_l_m, CI_r_m, SE_m,
              RMSE_m]
    results.append(values)

result_dir = pd.DataFrame(results, columns=['X-axis', 'Y-axis', 'p_f', 'R2_f', 'N_f', 'lower bound 95% CI female',
                                            'upper bound 95% CI female', 'Standard error female', 'RMSE_female', 'p_m',
                                            'R2_m', 'N_m', 'lower bound 95% CI male', 'upper bound 95% CI male',
                                            'Standard error male', 'RMSE_male'])
result_dir.to_csv(Results_path + '/04_Plots/ResultsPlots.csv', index=False)

# # boxplots of specific component weights
# MWF = df['Mineral weight fraction / -']
# OWF = df['Organic weight fraction / -']
# WWF = df['Water weight fraction / -']
# WF = [MWF, OWF, WWF]
#
# fig = plt.figure(figsize=(5.5, 4.5))
# ax1 = fig.add_subplot(111)
# bp = ax1.boxplot(WF)
# ax1.set_ylabel('Weight Fraction / -')
# ax1.set_xticklabels(['Mineral', 'Organic', 'Water'])
# plt.ylim(ymin=0)
# plt.savefig(os.path.join(Results_path + '/04_Plots/WF_boxplt.png'), dpi=300, bbox_inches='tight', format='png')
# # plt.show()
# plt.close()
#
# # boxplot of AMM/AMD
# AMM = df['Apparent Modulus Mineralized / GPa'].dropna().reset_index(drop=True)
# AMD = df['Apparent Modulus Demineralized / MPa'].dropna().reset_index(drop=True)
# AMM = AMM.values.tolist()
# AMD = AMD.values.tolist()
#
# # If we were to simply plot pts, we'd lose most of the interesting
# # details due to the outliers. So let's 'break' or 'cut-out' the y-axis
# # into two portions - use the top (ax) for the outliers, and the bottom
# # (ax2) for the details of the majority of our data
# f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
#
# # plot the same data on both axes
# ax.boxplot(AMM, positions=[1])
# ax2.boxplot(AMD, positions=[2])
#
# # zoom-in / limit the view to different portions of the data
# ax.set_ylim(10000, 20000)  # outliers only
# ax2.set_ylim(0, 300)  # most of the data
#
# # hide the spines between ax and ax2
# ax.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax.tick_params(labeltop=False, labelbottom=False)  # don't put tick labels at the top
# ax2.tick_params(labeltop=False)
# ax2.xaxis.tick_bottom()
# ax2.set_xticklabels(['Mineralized', 'Demineralized'])
#
# ax.set_ylabel('Apparent Modulus / MPa')
# ax.yaxis.set_label_coords(-0.12, 0)
#
# # This looks pretty good, and was fairly painless, but you can get that
# # cut-out diagonal lines look with just a bit more work. The important
# # thing to know here is that in axes coordinates, which are always
# # between 0-1, spine endpoints are at these locations (0,0), (0,1),
# # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
# # appropriate corners of each of our axes, and so long as we use the
# # right transform and disable clipping.
#
# d = .015  # how big to make the diagonal lines in axes coordinates
# # arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
# ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
# ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
#
# kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
#
# # What's cool about this is that now if we vary the distance between
# # ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# # the diagonal lines will move accordingly, and stay right at the tips
# # of the spines they are 'breaking'
# plt.savefig(os.path.join(Results_path + '/04_Plots/AM_boxplt.png'), dpi=300, bbox_inches='tight', format='png')
# # plt.show()
# plt.close()
