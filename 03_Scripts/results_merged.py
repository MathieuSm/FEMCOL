import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import butter,filtfilt
from scipy.signal import find_peaks
from scipy import stats
from pathlib import Path

Cwd = Path.cwd()
results_mineralized = pd.read_csv(str(Cwd / '04_Results/00_Mineralized/ResultsElasticTesting.csv'), skiprows=0)
results_demineralized = pd.read_csv(str(Cwd / '04_Results/01_Demineralized/ResultsFailureTesting.csv'), skiprows=0)
results_gravimetry = pd.read_csv(str(Cwd / '04_Results/02_Gravimetry/ResultsGravimetry.csv'), skiprows=0)
results_uCT = pd.read_csv(str(Cwd / '04_Results/03_uCT/ResultsUCT.csv'), skiprows=0)
results_raman = pd.read_csv(str(Cwd / '04_Results/05_Raman/RamanResults.csv'), skiprows=0)

results_merged = pd.DataFrame()
results_merged['Sample ID'] = results_mineralized['Sample ID']
results_merged['Age'] = results_gravimetry['age']
results_merged['Gender'] = results_gravimetry['sex']
results_merged['Stiffness Mineralized N/mm'] = results_mineralized['Stiffness N/mm']
results_merged['Apparent Modulus Mineralized MPa'] = results_mineralized['Apparent modulus MPa']
results_merged['Ultimate Force N'] = results_demineralized['Ultimate Force / N']
results_merged['Ultimate Stress MPa'] = results_demineralized['Ultimate stress / MPa']
results_merged['Ultimate Strain -'] = results_demineralized['Ultimate strain / -']
results_merged['Stiffness Demineralized N/mm'] = results_demineralized['Stiffness / N/mm']
results_merged['Apparent Modulus Demineralized MPa'] = results_demineralized['Apparent modulus / MPa']
results_merged['Density g/' + 'cm\u00B3'] = results_gravimetry['density / g/cm^3']
results_merged['Organic Weight g'] = results_gravimetry['organic weight / g']
results_merged['Mineral Weight g'] = results_gravimetry['mineral weight / g']
results_merged['Water Weight g'] = results_gravimetry['water weight / g']
results_merged['Mineral weight fraction -'] = results_gravimetry['weight fraction of mineral phase / -']
results_merged['Organic weight fraction -'] = results_gravimetry['weight fraction of organic phase / -']
results_merged['Water weight fraction -'] = results_gravimetry['weight fraction of water phase / -']
results_merged['Bone Volume Fraction -'] = results_uCT['Bone Volume Fraction -']
results_merged['Bone Mineral Density mg HA / ' + 'cm\u00B3'] = results_uCT['Bone Mineral Density mg HA / cm3']
results_merged['Tissue Mineral Density mg HA / ' + 'cm\u00B3'] = results_uCT['Tissue Mineral Density mg HA / cm3']
results_merged['Bone Mineral Content mg HA'] = results_uCT['Bone Mineral Content mg HA']
results_merged['Minimum Area ' + 'mm\u00B2'] = results_uCT['Min Area mm^2']
results_merged['Mean Apparent Area ' + 'mm\u00B2'] = results_uCT['Mean Apparent Area mm^2']
results_merged['Minimum Equivalent Diameter mm'] = results_uCT['Min Diameter mm']
results_merged['Mean Apparent Diameter mm'] = results_uCT['Mean Apparent Diameter mm']
results_merged['Mean Area Fraction -'] = results_uCT['Mean Area Fraction -']
results_merged['Min Area Fraction -'] = results_uCT['Min Area Fraction -']
results_merged['Mineral to Matrix Ratio -'] = results_raman['M2M_ratio_mean']
results_merged.to_csv(os.path.join('/home/stefan/Documents/PythonScripts/04_Results/', 'ResultsOverview.csv'),
                      index=False)


