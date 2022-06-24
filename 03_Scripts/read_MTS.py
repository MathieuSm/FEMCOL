import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import butter,filtfilt
from scipy.signal import find_peaks
from scipy import stats

def butter_lowpass_filter(data, cutoff, order=9):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


parent_dir = '/home/stefan/Documents/PythonScripts/02_Data/02_MTS/Elastic_testing_mineralized/'
filename_list = [File for File in os.listdir(parent_dir) if File.endswith('.csv')]
filename_list.sort()

result_dir = {}
for filename in filename_list:
    sample_ID = filename.split('/')[-1].split('_')[1]
    # load csv:
    df = pd.read_csv(parent_dir + filename, skiprows=2)
    df.rename(columns={'sec': 'time', 'N': 'force_MTS', 'N.1': 'force_lc', 'mm': 'disp_MTS', 'mm.1': 'disp_ext'},
              inplace=True)

    # peak detection:
    peaks_index, _ = find_peaks(df['force_MTS'], width=50)

    # linear regression:
    Indices = np.arange(peaks_index[-1], df.index[-1])
    Data = df.iloc[Indices[:int(len(Indices) / 3)]]

    # slope, intercept, r_value, p_value, std_err = stats.linregress(df['disp_ext'][peaks_index[-1]:], df['force_MTS'][peaks_index[-1]:])
    slope, intercept, r_value, p_value, std_err = stats.linregress(Data['disp_ext'], Data['force_MTS'])
    # x_last_cycle = np.array([df.iloc[-1]['disp_ext'], df['disp_ext'][peaks_index[-1]]])
    x_last_cycle = np.array([Data.iloc[0]['disp_ext'], Data.iloc[-1]['disp_ext']])


    plt.figure(figsize=(6, 4))
    plt.title(sample_ID)
    plt.plot(df['disp_ext'], df['force_MTS'], label='MTS')
    plt.plot(df['disp_ext'][peaks_index[-1]:], df['force_MTS'][peaks_index[-1]:])
    plt.plot(x_last_cycle, slope * x_last_cycle + intercept, 'k')
    plt.plot([], ' ', label=f'stiffness = {slope:.1f} N/mm')
    plt.ylabel('force / N')
    plt.xlabel('disp / mm')
    plt.legend()
    plt.show()
    # plt.savefig('', dpi=300)


    plt.plot(df['time'][peaks_index], df['force_MTS'][peaks_index], 'o')
    plt.show()

    # filter signals:
    fs = 102.4       # sample rate, Hz
    cutoff = 10
    nyq = 0.5 * fs
    df['force_lc_filtered'] = butter_lowpass_filter(df['force_lc'], cutoff)

    plt.figure()
    plt.plot(df['disp_ext'], df['force_lc'], label='MTS')
    plt.plot(df['disp_ext'], df['force_lc_filtered'], label='MTS')
    plt.ylabel('force / N')
    plt.xlabel('displacement ext / mm')
    plt.legend()
    plt.show()


    result_dir[sample_ID]['stiffness'] = slope
