import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import butter,filtfilt
from scipy.signal import find_peaks
from scipy import stats
from pathlib import Path

# definition of lowpass filter
def butter_lowpass_filter(data, cutoff, order=9):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# definition of path
Cwd = Path.cwd()
DataPath = Cwd / '02_Data/02_MTS/Failure_testing_demineralized/'
filename_list = [File for File in os.listdir(DataPath) if File.endswith('.csv')]
filename_list.sort()

i = 0
result = list()

sample_ID = '395L'
# load csv:
df = pd.read_csv(str(DataPath / '14_395L_fail.csv'), skiprows=2)
df.rename(columns={'sec': 'time', 'N': 'force_MTS', 'N.1': 'force_lc', 'mm': 'disp_MTS', 'mm.1': 'disp_ext'},
inplace = True)

# filter signals:
fs = 102.4  # sample rate, Hz
cutoff = 5
nyq = 0.5 * fs
df['force_lc_filtered'] = butter_lowpass_filter(df['force_lc'], cutoff)

# plot filtered and raw signal both at a time & save with respective sample_ID as name
plt.figure()
plt.title(sample_ID)
plt.plot(df['disp_ext'], df['force_lc'], label='raw')
plt.plot(df['disp_ext'], df['force_lc_filtered'], label='filtered')
plt.plot(df['disp_MTS'], df['force_MTS'], label='MTS raw')
plt.plot(df['disp_ext'], df['force_MTS'], label='disp_ext/force_MTS')
plt.plot(df['disp_MTS'], df['force_lc_filtered'], label='disp_MTS/force_lc_filtered')
plt.ylabel('force / N')
plt.xlabel('disp / mm')
plt.legend()
savepath = Cwd / '04_Results/01_Demineralized/00_force_disp/'
plt.savefig(os.path.join(savepath, 'force_disp_' + sample_ID + '.png'), dpi=300)
plt.show()

# calculate stress and strain, filter & put into dataframe
Pi = 3.1415
area = 2*2*Pi/4
l_initial = 6.5
stress = df['force_lc']/area
strain = df['disp_ext']/l_initial
df['stress_lc'] = stress
df['strain_ext'] = strain
df['stress_lc_filtered'] = butter_lowpass_filter(df['stress_lc'], cutoff)

# plot stress vs strain raw/filtered & save
plt.figure()
plt.title(sample_ID)
plt.plot(df['strain_ext'], df['stress_lc'], label='raw')
plt.plot(df['strain_ext'], df['stress_lc_filtered'], label='filtered')
plt.ylabel('stress / MPa')
plt.xlabel('strain / -')
plt.legend()
savepath = Cwd / '04_Results/01_Demineralized/01_stress_strain/'
plt.savefig(os.path.join(savepath, 'stress_strain_' + sample_ID + '.png'), dpi=300)
plt.show()

# find max value of stress_lc column
max_stress_raw = df['stress_lc'].max()

# locate index of max value
max_stress_index = df['stress_lc'][df['stress_lc'] == max_stress_raw].dropna()
max_stress_index = max_stress_index.index

# search for ultimate stress (filtered)
row_max_stress_filtered = df['stress_lc_filtered'].iloc[max_stress_index[0]]
ultimate_stress_filtered = round(row_max_stress_filtered, 2)

# search for ultimate strain
row_max_strain = df['strain_ext'].iloc[max_stress_index[0]]
ultimate_strain = round(row_max_strain, 4)

# find max value of force_lc column
max_force_raw = df['force_lc'].loc[max_stress_index[0]]

# search for ultimate force (raw)
max_force_index = max_stress_index[0]

# search for ultimate force (filtered)
row_max_force_filtered = df['force_lc_filtered'].iloc[max_force_index]
ultimate_force_filtered = round(row_max_force_filtered, 2)

## calculate values for last cycle of stress/strain curve only
# find peaks
peaks_index, _ = find_peaks(df['force_lc'], width=200)

# define range between last peak and second last peak to subsequently search for stress minimum as start point
start_range_strain = df['strain_ext'][peaks_index[7]:max_stress_index[0]]
start_range_stress = df['stress_lc_filtered'][peaks_index[7]:max_stress_index[0]]
start_range = pd.DataFrame()
start_range['strain_ext'] = start_range_strain
start_range['stress_lc_filtered'] = start_range_stress

# find smallest stress value & corresponding index as starting point
start_value = min(start_range['stress_lc_filtered'])
start_ind = start_range.loc[start_range['stress_lc_filtered'] == start_value]
start_index = start_ind.index

# find max stress value & corresponding index as end point
end_value = df['stress_lc_filtered'][max_stress_index[0]]
end_ind = df.loc[df['stress_lc_filtered'] == end_value]
end_index = end_ind.index

# create dataframe of final cycle using start/end index
last_cycle_stress = df['stress_lc_filtered'].iloc[start_index[0]:end_index[0]]
last_cycle_strain = df['strain_ext'].iloc[start_index[0]:end_index[0]]
last_cycle_strain = last_cycle_strain.dropna().reset_index(drop=True)
last_cycle_stress = last_cycle_stress.dropna().reset_index(drop=True)

last_cycle = pd.DataFrame()
last_cycle['last_cycle_strain'] = round(last_cycle_strain, 5)
last_cycle['last_cycle_stress'] = round(last_cycle_stress, 5)

plt.plot(df['strain_ext'], df['stress_lc_filtered'], label='filtered')
plt.plot(last_cycle['last_cycle_strain'], last_cycle['last_cycle_stress'], color='green', label='last cycle')
plt.scatter(df['strain_ext'][peaks_index], df['stress_lc_filtered'][peaks_index], marker='o', color='red',
            label='peaks')
plt.scatter(last_cycle['last_cycle_strain'][len(last_cycle)-1], last_cycle['last_cycle_stress'][len(last_cycle)-1],
            marker='o', color='black', label='ultimate point')
plt.title(sample_ID)
plt.ylabel('stress / MPa')
plt.xlabel('strain / -')
plt.legend()
plt.show()
# plt.close()

## calculate apparent modulus by using rolling regression
# definition of strain region where regression should be carried out
upper_strain = max(last_cycle['last_cycle_strain'])
lower_strain = last_cycle['last_cycle_strain'].iloc[0]

# window width approx. 1/3 of curve length
window_width = round(1/3*(end_index[0] - start_index[0]))
slope_values = list()

# rolling linear regression
for x in range(0, len(last_cycle) - 1 - window_width + 1, 1):
    last_cycle_mod = last_cycle[x:x + window_width]
    slope, intercept, r_value, p_value, std_err = stats.linregress(last_cycle_mod['last_cycle_strain'],
                                                                   last_cycle_mod['last_cycle_stress'])
    slope_value = slope
    slope_values.append(slope_value)
apparent_modulus = round(max(slope_values), 2)

# calculate stiffness
# create dataframe of final cycle using start/end index
last_cycle_force = df['force_lc_filtered'].iloc[start_index[0]:end_index[0]]
last_cycle_disp = df['disp_ext'].iloc[start_index[0]:end_index[0]]
last_cycle_disp = last_cycle_disp.dropna().reset_index(drop=True)
last_cycle_force = last_cycle_force.dropna().reset_index(drop=True)

last_cycle_fd = pd.DataFrame()
last_cycle_fd['last_cycle_disp'] = round(last_cycle_disp, 5)
last_cycle_fd['last_cycle_force'] = round(last_cycle_force, 5)

plt.plot(df['disp_ext'], df['force_lc_filtered'], label='filtered')
plt.plot(last_cycle_fd['last_cycle_disp'], last_cycle_fd['last_cycle_force'], color='green', label='last cycle')
plt.scatter(df['disp_ext'][peaks_index], df['force_lc_filtered'][peaks_index], marker='o', color='red',
            label='peaks')
plt.scatter(last_cycle_fd['last_cycle_disp'][len(last_cycle_fd)-1], last_cycle_fd['last_cycle_force'][len(
    last_cycle_fd)-1], marker='o', color='black', label='ultimate point')
plt.title(sample_ID + '_stiffness')
plt.ylabel('force / N')
plt.xlabel('disp / mm')
plt.legend()
plt.show()
# plt.close()

## calculate stiffness by using rolling regression
# definition of disp region where regression should be carried out
upper_disp = max(last_cycle_fd['last_cycle_disp'])
lower_disp = last_cycle_fd['last_cycle_disp'].iloc[0]

slope_values_stiff = list()

# rolling linear regression
for j in range(0, len(last_cycle_fd) - 1 - window_width + 1, 1):
    last_cycle_mod = last_cycle_fd[j:j + window_width]
    stiff, intercept, r_value, p_value, std_err = stats.linregress(last_cycle_mod['last_cycle_disp'],
                                                                   last_cycle_mod['last_cycle_force'])
    slope_value_stiff = stiff
    slope_values_stiff.append(slope_value_stiff)
stiffness = round(max(slope_values_stiff),2)

values = [sample_ID, ultimate_stress_filtered, ultimate_strain, ultimate_force_filtered, apparent_modulus, stiffness]

result.append(values)
result_dir = pd.DataFrame(result, columns=['Sample ID', 'Ultimate stress / MPa', 'Ultimate strain / -',
                                           'Ultimate Force / N', 'Apparent modulus / MPa', 'Stiffness / N/mm'])

result_dir.to_csv(os.path.join('/home/stefan/Documents/PythonScripts/04_Results/01_Demineralized/',
                               'ResultsFailureTesting395L.csv'), index=False)

print(result_dir)

