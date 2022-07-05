import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import butter,filtfilt
from scipy.signal import find_peaks
from scipy import stats
from pathlib import Path


# Data = pd.read_csv(str(DataPath / 'pilot_polymer_failure.csv'),header=2)
# Data.columns = ['Time [s]', 'Axial Force [N]',
#                 'Load cell [N]', 'Axial Displacement [mm]',
#                 'MTS Displacement [mm]', 'Axial Count [cycles]']

# definition of lowpass filter
def butter_lowpass_filter(data, cutoff, order=9):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

Cwd = Path.cwd()
DataPath = Cwd / '02_Data/02_MTS/Elastic_testing_mineralized/'
parent_dir = '/home/stefan/Documents/PythonScripts/02_Data/02_MTS/Elastic_testing_mineralized/'
# filename_list = [File for File in os.listdir(parent_dir) if File.endswith('.csv')]
filename_list = [File for File in os.listdir(DataPath) if File.endswith('.csv')]
filename_list.sort()

result = list()
i = 0

for filename in filename_list:
    sample_ID = filename.split('/')[-1].split('_')[1]
    # load csv:
    df = pd.read_csv(str(DataPath / filename_list[i]), skiprows = 2)
    df.rename(columns={'sec': 'time', 'N': 'force_MTS', 'N.1': 'force_lc', 'mm': 'disp_MTS', 'mm.1': 'disp_ext'},
    inplace = True)
    i = i + 1

    # filter signals:
    fs = 102.4       # sample rate, Hz
    cutoff = 10
    nyq = 0.5 * fs
    df['force_MTS_filtered'] = butter_lowpass_filter(df['force_MTS'], cutoff)

    # plot filtered signals (displacement extensometer vs. force MTS)
    plt.figure()
    plt.title(sample_ID)
    plt.plot(df['disp_ext'], df['force_MTS'], label='MTS raw')
    plt.plot(df['disp_ext'], df['force_MTS_filtered'], label='MTS filtered')
    plt.ylabel('force MTS / N')
    plt.xlabel('disp ext / mm')
    plt.legend()
    plt.show()

    # peak detection:
    peaks_index, _ = find_peaks(df['force_MTS'], width=50)

    # linear regression:
    Indices = np.arange(peaks_index[-1], df.index[-1])
    Data_reg = df.iloc[Indices[0:int(len(Indices) / 3)]]

    # slope, intercept, r_value, p_value, std_err = stats.linregress(df['disp_ext'][peaks_index[-1]:], df['force_MTS'][peaks_index[-1]:])
    slope, intercept, r_value, p_value, std_err = stats.linregress(Data_reg['disp_ext'], Data_reg['force_MTS'])
    # x_last_cycle = np.array([df.iloc[-1]['disp_ext'], df['disp_ext'][peaks_index[-1]]])
    x_last_cycle = np.array([Data_reg.iloc[0]['disp_ext'], Data_reg.iloc[-1]['disp_ext']])

    # generate plot
    plt.figure(figsize=(6, 4))
    plt.title(sample_ID)
    plt.plot(df['disp_ext'], df['force_MTS'], label='MTS')
    plt.plot(df['disp_ext'][peaks_index[-1]:], df['force_MTS'][peaks_index[-1]:])
    plt.plot(x_last_cycle, slope * x_last_cycle + intercept, 'k')
    plt.plot([], ' ', label=f'stiffness = {slope:.0f} N/mm')
    plt.ylabel('force MTS / N')
    plt.xlabel('disp ext / mm')
    plt.legend()
    plt.show()
    # plt.savefig('', dpi=300)

    # plt.plot(df['time'][peaks_index], df['force_MTS'][peaks_index], 'o')
    # plt.show()

    # create list with current values which are sample_ID & slope & add them to result list which is then converted
    # to dataframe
    values = [sample_ID, round(slope)]
    result.append(values)
    result_dir = pd.DataFrame(result, columns=['Sample ID', 'Stiffness N/mm'])

print(result_dir)

# safe dataframe to csv
result_dir.to_csv(os.path.join('/home/stefan/Documents/PythonScripts/02_Data/02_MTS/', 'ResultsElasticTesting.csv'), index=False)

