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

Cwd = Path.cwd()
DataPath = Cwd / '02_Data/02_MTS/Failure_testing_demineralized/'
filename_list = [File for File in os.listdir(DataPath) if File.endswith('.csv')]
filename_list.sort()

i = 0
result = list()

for filename in filename_list:
    sample_ID = filename.split('/')[-1].split('_')[1]
    # load csv:
    df = pd.read_csv(str(DataPath / filename_list[i]), skiprows = 2)
    df.rename(columns={'sec': 'time', 'N': 'force_MTS', 'N.1': 'force_lc', 'mm': 'disp_MTS', 'mm.1': 'disp_ext'},
    inplace = True)
    i = i + 1

    # filter signals:
    fs = 102.4  # sample rate, Hz
    cutoff = 5
    nyq = 0.5 * fs
    df['force_lc_filtered'] = butter_lowpass_filter(df['force_lc'], cutoff)

    # plot filtered and raw signal both at a time & save with respective sample_ID as name
    # plt.figure()
    # plt.title(sample_ID)
    # plt.plot(df['disp_ext'], df['force_lc'], label='raw')
    # plt.plot(df['disp_ext'], df['force_lc_filtered'], label='filtered')
    # plt.ylabel('force lc / N')
    # plt.xlabel('disp ext / mm')
    # plt.legend()
    # plt.close()

    # calculate stress and strain, filter & put into dataframe
    Pi = 3.1415
    area = 2*2*Pi/4
    l_initial = 6.5
    stress = df['force_lc']/area
    strain = df['disp_ext']/l_initial
    df['stress_lc'] = stress
    df['strain_ext'] = strain
    df['stress_lc_filtered'] = butter_lowpass_filter(df['stress_lc'], cutoff)

    # plot stress vs strain raw/filtered
    plt.figure()
    plt.title(sample_ID)
    plt.plot(df['strain_ext'], df['stress_lc'], label='raw')
    plt.plot(df['strain_ext'], df['stress_lc_filtered'], label='filtered')
    plt.ylabel('stress / MPa')
    plt.xlabel('strain / -')
    plt.legend()
    savepath = Cwd / '02_Data/02_MTS/'
    plt.savefig(os.path.join(savepath, 'failure_' + sample_ID + '.png'), dpi=300)

    # find max value of stress_MTS column
    column = df['stress_lc']
    max_stress_raw = column.max()

    # locate index of max value
    row_max_stress = df.loc[df['stress_lc'] == max_stress_raw]
    max_stress_index = row_max_stress.index

    # search for ultimate stress (filtered)
    row_max_stress_filtered = df.iloc[max_stress_index]
    # ultimate_stress_filtered = row_max_stress_filtered['stress_lc_filtered']
    # returns only number as type 'array'
    ultimate_stress_filtered = np.round(row_max_stress_filtered['stress_lc_filtered'].values, 2)

    # search for ultimate strain
    row_max_strain = df.iloc[max_stress_index]
    ultimate_strain = np.round(row_max_strain['strain_ext'].values, 4)

    values = [sample_ID, ultimate_stress_filtered, ultimate_strain]
    result.append(values)
    result_dir = pd.DataFrame(result, columns=['Sample ID', 'Ultimate stress MPa', 'Ultimate strain -'])

print(result_dir)
