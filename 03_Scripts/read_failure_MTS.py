import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import butter, filtfilt
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
results_uCT = pd.read_csv(str('/home/stefan/Documents/PythonScripts/04_Results/03_uCT/ResultsUCT.csv'), skiprows=0)
results_uCT = results_uCT.drop(index=[8, 13, 20, 24, 27, 37], axis=0)
results_uCT = results_uCT.reset_index(drop=True)

i = 0
counter = 0
result = list()

# loop over .csv files in Folder
for filename in filename_list:
    sample_ID = filename.split('/')[-1].split('_')[1]
    # load csv:
    df = pd.read_csv(str(DataPath / filename_list[i]), skiprows=2)
    df.rename(columns={'sec': 'time', 'N': 'force_MTS', 'N.1': 'force_lc', 'mm': 'disp_MTS', 'mm.1': 'disp_ext'},
    inplace = True)
    i = i + 1

    # filter signals:
    fs = 102.4  # sample rate, Hz
    cutoff = 5
    nyq = 0.5 * fs
    df['force_lc_filtered'] = butter_lowpass_filter(df['force_lc'], cutoff)

    # calculate stress and strain, filter & put into dataframe
    Pi = 3.14159265
    l_initial = 6.5
    area = results_uCT['min_Area'][counter]
    stress = df['force_lc'] / area
    strain = df['disp_ext'] / l_initial
    df['stress_lc'] = stress
    df['strain_ext'] = strain
    df['stress_lc_filtered'] = butter_lowpass_filter(df['stress_lc'], cutoff)
    counter = counter + 1

    # find max value of stress_lc column
    column = df['stress_lc']
    max_stress_raw = column.max()

    # locate index of max value
    row_max_stress = df.loc[df['stress_lc'] == max_stress_raw]
    max_stress_index = row_max_stress.index

    # search for ultimate stress (filtered)
    row_max_stress_filtered = df.iloc[max_stress_index]
    ultimate_stress_filtered = np.round(row_max_stress_filtered['stress_lc_filtered'].values, 2)
    ultimate_stress_filtered = ultimate_stress_filtered[0]

    # search for ultimate strain
    row_max_strain = df.iloc[max_stress_index]
    ultimate_strain = np.round(row_max_strain['strain_ext'].values, 4)
    ultimate_strain = ultimate_strain[0]

    # find max value of force_lc column
    column_force = df['force_lc']
    max_force_raw = column_force.max()

    # search for ultimate force (raw)
    row_max_force = df.loc[df['force_lc'] == max_force_raw]
    max_force_index = row_max_force.index

    # search for ultimate force (filtered)
    row_max_force_filtered = df.iloc[max_force_index]
    ultimate_force_filtered = np.round(row_max_force_filtered['force_lc_filtered'].values, 2)
    ultimate_force_filtered = ultimate_force_filtered[0]

    ## calculate values for last unloading cycle of stress/strain curve
    # find peaks
    peaks_index, _ = find_peaks(df['force_lc'], width=200)

    # define range between last peak and second last peak to subsequently search for stress minimum as start point
    start_range_strain_unload = df['strain_ext'][peaks_index[6]:peaks_index[7]]
    start_range_stress_unload = df['stress_lc_filtered'][peaks_index[6]:peaks_index[7]]
    start_range_unload = pd.DataFrame()
    start_range_unload['strain_ext'] = start_range_strain_unload
    start_range_unload['stress_lc_filtered'] = start_range_stress_unload

    # find smallest stress value & corresponding index as starting point
    start_value_unload = min(start_range_unload['stress_lc_filtered'])
    start_ind_unload = start_range_unload.loc[start_range_unload['stress_lc_filtered'] == start_value_unload]
    start_index_unload = start_ind_unload.index

    # create dataframe using unloading part of last cycle
    last_cycle_unloading = pd.DataFrame()
    last_cycle_unloading['last_cycle_strain'] = df['strain_ext'].iloc[start_index_unload[0]:peaks_index[7]]
    last_cycle_unloading['last_cycle_stress'] = df['stress_lc_filtered'].iloc[start_index_unload[0]:peaks_index[7]]

    ## calculate stiffness using last unloading cycle
    # create dataframe of final unloading cycle using start/end index
    last_cycle_force_unload = df['force_lc_filtered'].iloc[start_index_unload[0]:peaks_index[7]]
    last_cycle_disp_unload = df['disp_ext'].iloc[start_index_unload[0]:peaks_index[7]]
    last_cycle_disp_unload = last_cycle_disp_unload.dropna().reset_index(drop=True)
    last_cycle_force_unload = last_cycle_force_unload.dropna().reset_index(drop=True)

    last_cycle_fd_unload = pd.DataFrame()
    last_cycle_fd_unload['last_cycle_disp_unload'] = round(last_cycle_disp_unload, 5)
    last_cycle_fd_unload['last_cycle_force_unload'] = round(last_cycle_force_unload, 5)

    # calculate stiffness of last unloading cycle (preconditioning) by using rolling regression
    window_width_unload = round(1 / 3 * (last_cycle_fd_unload.index[-1]))
    slope_values_stiff_unload = list()
    intercept_values_stiff_unload = list()

    # rolling linear regression
    for k in range(0, len(last_cycle_fd_unload) - 1 - window_width_unload + 1, 1):
        last_cycle_mod_unload = last_cycle_fd_unload[k:k + window_width_unload]
        slope_stiff_unload, intercept_stiff_unload, r_value, p_value, std_err = stats.linregress(
            last_cycle_mod_unload['last_cycle_disp_unload'], last_cycle_mod_unload['last_cycle_force_unload'])
        # collect slope value & add to growing list; same for intercept
        slope_value_stiff_unload = slope_stiff_unload
        slope_values_stiff_unload.append(slope_value_stiff_unload)
        intercept_value_stiff_unload = intercept_stiff_unload
        intercept_values_stiff_unload.append(intercept_value_stiff_unload)

    # Create DataFrames for slope/stiffness to search for max. values & corresponding indices; create cycle for plotting
    slope_value_stiff_unload_df = pd.DataFrame(slope_values_stiff_unload)
    intercept_values_stiff_unload_df = pd.DataFrame(intercept_values_stiff_unload)
    stiffness = slope_value_stiff_unload_df.max()[0]
    max_slope_index_stiff_unload = slope_value_stiff_unload_df[slope_value_stiff_unload_df == stiffness].dropna()
    max_slope_index_stiff_unload = max_slope_index_stiff_unload.index
    intercept_value_max_stiff_unload = intercept_values_stiff_unload_df.loc[max_slope_index_stiff_unload[0]].values[0]
    last_cycle_unload_plot = last_cycle_fd_unload[max_slope_index_stiff_unload[0]:max_slope_index_stiff_unload[0] + window_width_unload]

    # generate plot
    plt.figure(figsize=(6, 4))
    plt.title(sample_ID)
    plt.plot(df['disp_ext'][0:peaks_index[-1]], df['force_lc_filtered'][0:peaks_index[-1]], label='filtered')
    plt.plot(last_cycle_fd_unload['last_cycle_disp_unload'], last_cycle_fd_unload['last_cycle_force_unload'],
             label='last unloading cycle')
    plt.plot(last_cycle_unload_plot['last_cycle_disp_unload'], last_cycle_unload_plot['last_cycle_force_unload'],
             label='regress area', color='k')
    plt.plot(last_cycle_unload_plot['last_cycle_disp_unload'], stiffness *
             last_cycle_unload_plot['last_cycle_disp_unload'] + intercept_value_max_stiff_unload, 'r', label='fit')
    plt.plot([], ' ', label=f'stiffness = {stiffness:.0f} N/mm')
    plt.ylabel('force / N')
    plt.xlabel('disp / mm')
    plt.legend()
    savepath = Cwd / '04_Results/00_Mineralized/00_force_disp/'
    plt.savefig(os.path.join(savepath, 'force_disp_' + sample_ID + '.png'), dpi=300)
    plt.show()
    # plt.close()

    ## calculate apparent modulus using last unloading cycle
    # create dataframe of final unloading cycle using start/end index
    last_cycle_stress_unload = df['stress_lc_filtered'].iloc[start_index_unload[0]:peaks_index[7]]
    last_cycle_strain_unload = df['strain_ext'].iloc[start_index_unload[0]:peaks_index[7]]
    last_cycle_strain_unload = last_cycle_strain_unload.dropna().reset_index(drop=True)
    last_cycle_stress_unload = last_cycle_stress_unload.dropna().reset_index(drop=True)

    last_cycle_ss_unload = pd.DataFrame()
    last_cycle_ss_unload['last_cycle_strain_unload'] = round(last_cycle_strain_unload, 5)
    last_cycle_ss_unload['last_cycle_stress_unload'] = round(last_cycle_stress_unload, 5)

    window_width_unload = round(1 / 3 * len(last_cycle_ss_unload))
    slope_values_app_unload = list()
    intercept_values_app_unload = list()

    # rolling linear regression to calculate apparent modulus of last unloading cycle
    for k in range(0, len(last_cycle_ss_unload) - 1 - window_width_unload + 1, 1):
        last_cycle_mod_unload = last_cycle_ss_unload[k:k + window_width_unload]
        slope_app_unload, intercept_app_unload, r_value, p_value, std_err = stats.linregress(
            last_cycle_mod_unload['last_cycle_strain_unload'],
            last_cycle_mod_unload['last_cycle_stress_unload'])
        # collect slope value & add to growing list; same for intercept
        slope_value_app_unload = slope_app_unload
        slope_values_app_unload.append(slope_value_app_unload)
        intercept_value_app_unload = intercept_app_unload
        intercept_values_app_unload.append(intercept_value_app_unload)

    # Create DataFrames for slope/app. modulus to search for max. values & corresponding indices; create cycle for plotting
    slope_value_app_unload_df = pd.DataFrame(slope_values_app_unload)
    intercept_values_app_unload_df = pd.DataFrame(intercept_values_app_unload)
    apparent_modulus = slope_value_app_unload_df.max()[0]
    max_slope_index_app_unload = slope_value_app_unload_df[slope_value_app_unload_df == apparent_modulus].dropna()
    max_slope_index_app_unload = max_slope_index_app_unload.index
    intercept_value_max_app_unload = intercept_values_app_unload_df.loc[max_slope_index_app_unload[0]].values[0]
    last_cycle_unload_plot = last_cycle_ss_unload[max_slope_index_app_unload[0]:max_slope_index_app_unload[0] + window_width_unload]

    # generate plot
    plt.figure(figsize=(6, 4))
    plt.title(sample_ID)
    plt.plot(df['strain_ext'][0:peaks_index[-1]], df['stress_lc_filtered'][0:peaks_index[-1]], label='filtered')
    plt.plot(last_cycle_ss_unload['last_cycle_strain_unload'], last_cycle_ss_unload['last_cycle_stress_unload'],
             label='last unloading cycle')
    plt.plot(last_cycle_unload_plot['last_cycle_strain_unload'], last_cycle_unload_plot['last_cycle_stress_unload'],
             label='regress area', color='k')
    plt.plot(last_cycle_unload_plot['last_cycle_strain_unload'], apparent_modulus *
             last_cycle_unload_plot['last_cycle_strain_unload'] + intercept_value_max_app_unload, 'r', label='fit')
    plt.plot([], ' ', label=f'apparent modulus = {apparent_modulus:.0f} MPa')
    plt.ylabel('stress / MPa')
    plt.xlabel('strain / -')
    plt.legend()
    savepath = Cwd / '04_Results/01_Demineralized/01_stress_strain/'
    plt.savefig(os.path.join(savepath, 'stress_strain_' + sample_ID + '.png'), dpi=300)
    plt.show()
    # plt.close()

    # collect all data in list
    values = [sample_ID, ultimate_stress_filtered, ultimate_strain, ultimate_force_filtered, apparent_modulus,
              stiffness]

    # update list with each iteration's data & generate dataframe
    result.append(values)
    result_dir = pd.DataFrame(result, columns=['Sample ID', 'Ultimate stress / MPa', 'Ultimate strain / -',
                                               'Ultimate Force / N', 'Apparent modulus / MPa', 'Stiffness / N/mm'])

# load manually adjusted curve from sample 395L (dataframe)
df1 = pd.read_csv(str('/home/stefan/Documents/PythonScripts/04_Results/01_Demineralized/ResultsFailureTesting395L.csv'),
                  skiprows=0)
# initialize new dataframe
result_dir_new = pd.DataFrame()

# copy last column of existing dataframe into new dataframe as "clipboard"
result_dir_new['Stiffness / N/mm'] = result_dir['Stiffness / N/mm']

# replace data of existing dataframe by manually adjusted data
result_dir.loc[12] = df1.loc[0]

# paste last column from "clipboard" into existing dataframe
result_dir['Stiffness / N/mm'] = result_dir_new['Stiffness / N/mm']

missing_sample_IDs = pd.DataFrame({'Sample ID': ['390R', '395R', '400R', '402L', '403R', '433L']})
# missing_IDs = pd.DataFrame(missing_sample_IDs, columns=['Sample ID'])
result_dir = pd.concat([result_dir, missing_sample_IDs])
result_dir_sorted = result_dir.sort_values(by=['Sample ID'], ascending=True)

result_dir_sorted.to_csv(os.path.join('/home/stefan/Documents/PythonScripts/04_Results/01_Demineralized/',
                               'ResultsFailureTesting.csv'), index=False)

