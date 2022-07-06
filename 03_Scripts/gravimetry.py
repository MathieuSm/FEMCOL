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
    density = df['wet weight']*H20_density/(df['wet weight']-df['H2O weight'])
Sample_ID = df['Sample ID']
results = [Sample_ID, density]
result_dir = pd.DataFrame(results, columns=['Sample ID','density g/cm^3'])

print(result_dir)
