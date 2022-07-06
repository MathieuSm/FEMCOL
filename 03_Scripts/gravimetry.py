import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import butter,filtfilt
from scipy.signal import find_peaks
from scipy import stats
from pathlib import Path

# specify file path
Cwd = Path.cwd()
DataPath = Cwd / '02_Data/03_Scale/'
FileName = 'Gravimetric_analysis.xlsx'

# import xlsx file using pandas
df = pd.read_excel(str(DataPath / FileName))

# density of distilled water at 23°C
H20_density = 0.99754

# calculate density with density=wet weight*density of H2O/(wet weight-H2O weight)
for row in range(len(df)):
    density = df['wet weight'] * H20_density / (df['wet weight'] - df['H2O weight'])
    w_organic = df['dry weight'] - df['ash weight']
    w_mineral = df['ash weight']
    w_water = df['wet weight'] - df['dry weight']
    wf_mineral = df['ash weight'] / df['wet weight']
    wf_organic = w_organic / df['wet weight']
    wf_water = 1 - wf_organic - wf_mineral

#
result_dir = pd.DataFrame()
result_dir['Sample ID'] = df['Sample ID']
result_dir['density / g/cm^3'] = density
result_dir['organic weight / g'] = w_organic
result_dir['mineral weight / g'] = w_mineral
result_dir['water weight / g'] = w_water
result_dir['weight fraction of mineral phase / -'] = wf_mineral
result_dir['weight fraction of organic phase / -'] = wf_organic
result_dir['weight fraction of water phase / -'] = wf_water

# results = [Sample_ID, density]
# result_dir = pd.DataFrame(results, columns=['Sample ID','density g/cm^3'])

result_dir.to_csv(os.path.join('/home/stefan/Documents/PythonScripts/04_Results/02_Gravimetry/', 'ResultsGravimetry.csv'), index=False)

print(result_dir)


