import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import butter,filtfilt
from scipy.signal import find_peaks
from scipy import stats
from pathlib import Path

results_mineralized = pd.read_csv(str('/home/stefan/Documents/PythonScripts/04_Results/00_Mineralized/ResultsElasticTesting.csv'),
                  skiprows=0)
results_demineralized = pd.read_csv(str('/home/stefan/Documents/PythonScripts/04_Results/01_Demineralized/ResultsFailureTesting.csv'),
                  skiprows=0)
results_gravimetry = pd.read_csv(str('/home/stefan/Documents/PythonScripts/04_Results/02_Gravimetry/ResultsGravimetry.csv'),
                  skiprows=0)
results_uCT = pd.read_csv(str('/home/stefan/Documents/PythonScripts/04_Results/03_uCT/ResultsUCT.csv'), skiprows=0)

age = [87, 87, 76, 81, 88, 88, 91, 91, 96, 94, 94, 78, 78, 91, 91, 64, 64, 71, 85, 85, 77, 77, 90, 90, 80, 80, 89, 89,
       89, 89, 78, 79, 96, 74, 74, 92, 65, 88, 88, '']
gender = ['F', 'F', 'F', 'F', 'M', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'M', 'M', 'F', 'M', 'M', 'F', 'F',
          'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'F', 'F', '']
age_df = pd.DataFrame(age, columns=['Age'])
gender_df = pd.DataFrame(gender, columns=['Gender'])

results_merged = pd.DataFrame()
results_merged['Sample ID'] = results_mineralized['Sample ID']
results_merged['Age'] = age_df
results_merged['Gender'] = gender_df
results_merged['Stiffness Mineralized N/mm'] = results_mineralized['Stiffness N/mm']
results_merged['Apparent Modulus Mineralized MPa'] = results_mineralized['Apparent modulus MPa']
results_merged['Ultimate Force N'] = results_demineralized['Ultimate Force / N']
results_merged['Ultimate Stress MPa'] = results_demineralized['Ultimate stress / MPa']
results_merged['Ultimate Strain -'] = results_demineralized['Ultimate strain / -']
results_merged['Stiffness Demineralized N/mm'] = results_demineralized['Stiffness / N/mm']
results_merged['Stiffness Demineralized Unloading N/mm'] = results_demineralized['Stiffness unloading / N/mm']
results_merged['Apparent Modulus Demineralized MPa'] = results_demineralized['Apparent modulus / MPa']
results_merged['Density g/cm^3'] = results_gravimetry['density / g/cm^3']
results_merged['Organic Weight g'] = results_gravimetry['organic weight / g']
results_merged['Mineral Weight g'] = results_gravimetry['mineral weight / g']
results_merged['Water Weight g'] = results_gravimetry['water weight / g']
results_merged['Mineral weight fraction -'] = results_gravimetry['weight fraction of mineral phase / -']
results_merged['Organic weight fraction -'] = results_gravimetry['weight fraction of organic phase / -']
results_merged['Water weight fraction -'] = results_gravimetry['weight fraction of water phase / -']
results_merged['Bone Volume Fraction -'] = results_uCT['Bone Volume Fraction -']
results_merged['Bone Mineral Density mg HA / cm^3'] = results_uCT['Bone Mineral Density mg HA / cm3']
results_merged['Tissue Mineral Density mg HA / cm^3'] = results_uCT['Tissue Mineral Density mg HA / cm3']
results_merged['Bone Mineral Content mg HA'] = results_uCT['Bone Mineral Content mg HA']

results_merged.to_csv(os.path.join('/home/stefan/Documents/PythonScripts/04_Results/', 'ResultsOverview.csv'),
                      index=False)

