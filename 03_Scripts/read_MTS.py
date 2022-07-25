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


Cwd = Path.cwd()
DataPath = Cwd / '02_Data/02_MTS/Elastic_testing_mineralized/'
filename_list = [File for File in os.listdir(DataPath) if File.endswith('.csv')]
filename_list.sort()
results_uCT = pd.read_csv(str('/home/stefan/Documents/PythonScripts/04_Results/03_uCT/ResultsUCT.csv'), skiprows=0)
results_uCT = results_uCT.drop(index=[8, 13, 20, 24, 37], axis=0)
results_uCT = results_uCT.reset_index(drop=True)


result = list()
i = 0
counter = 0

for filename in filename_list:
    sample_ID = filename.split('/')[-1].split('_')[1]
    # load csv:
    df = pd.read_csv(str(DataPath / filename_list[i]), skiprows=2)
    df.rename(columns={'sec': 'time', 'N': 'force_MTS', 'N.1': 'force_lc', 'mm': 'disp_MTS', 'mm.1': 'disp_ext'},
              inplace=True)
    i = i + 1

    # filter signals:
    fs = 102.4  # sample rate, Hz
    cutoff = 5
    nyq = 0.5 * fs
    df['force_lc_filtered'] = butter_lowpass_filter(df['force_lc'], cutoff)

    # plot filtered signals (displacement extensometer vs. force MTS)
    plt.figure()
    plt.title(sample_ID)
    plt.plot(df['disp_ext'], df['force_lc'], label='raw')
    plt.plot(df['disp_ext'], df['force_lc_filtered'], label='filtered')
    plt.ylabel('force lc / N')
    plt.xlabel('disp ext / mm')
    plt.legend()
    # plt.show()
    plt.close()

    # peak detection
    peaks_index, _ = find_peaks(df['force_lc'], width=50)

    # calculate values for last cycle of force/disp curve only
    last_cycle_force = df['force_lc_filtered'][peaks_index[-1]:]
    last_cycle_disp = df['disp_ext'][peaks_index[-1]:]
    last_cycle_disp = last_cycle_disp.dropna().reset_index(drop=True)
    last_cycle_force = last_cycle_force.dropna().reset_index(drop=True)

    # isolate stress/strain values
    last_cycle = pd.DataFrame()
    last_cycle['last_cycle_force'] = round(last_cycle_force, 5)
    last_cycle['last_cycle_disp'] = round(last_cycle_disp, 5)

    upper_disp = max(last_cycle['last_cycle_disp'])
    lower_disp = 0.005

    # problem here
    # identify index range of defined region
    upper_cond_disp = last_cycle.loc[last_cycle['last_cycle_disp'] == upper_disp]
    lower_cond_disp = last_cycle.loc[last_cycle['last_cycle_disp'] >= lower_disp]
    max_disp_ind = min(upper_cond_disp.index)
    min_disp_ind = max(lower_cond_disp.index)
    last_cycle = last_cycle[max_disp_ind:min_disp_ind]

    window_width = round(1 / 3 * (last_cycle.index[-1]))

    # rolling linear regression for stiffness calculation
    slope_values_stiff = list()
    intercept_values_stiff = list()
    for x in range(0, len(last_cycle) - 1 - window_width + 1, 1):
        last_cycle_mod = last_cycle[x:x + window_width]
        slope_stiff, intercept_stiff, r_value, p_value, std_err = stats.linregress(last_cycle_mod['last_cycle_disp'],
                                                                       last_cycle_mod['last_cycle_force'])
        slope_value_stiff = slope_stiff
        intercept_value = intercept_stiff
        slope_values_stiff.append(slope_value_stiff)
        intercept_values_stiff.append(intercept_value)
    slope_value_df = pd.DataFrame(slope_values_stiff)
    intercept_values_df = pd.DataFrame(intercept_values_stiff)
    stiffness = slope_value_df.max()[0]
    max_slope_index = slope_value_df[slope_value_df == stiffness].dropna()
    max_slope_index = max_slope_index.index
    intercept_value_max_stiff = intercept_values_df.loc[max_slope_index[0]].values[0]
    last_cycle_plot = last_cycle[max_slope_index[0]:max_slope_index[0] + window_width]

    # generate plot
    plt.figure(figsize=(6, 4))
    plt.title(sample_ID)
    plt.plot(df['disp_ext'], df['force_lc_filtered'], label='filtered')
    plt.plot(last_cycle['last_cycle_disp'], last_cycle['last_cycle_force'], label='last unloading cycle 0.005-0.015')
    plt.plot(last_cycle_plot['last_cycle_disp'], last_cycle_plot['last_cycle_force'], label='regress area',
             color='k')
    plt.plot(last_cycle_plot['last_cycle_disp'], stiffness * last_cycle_plot['last_cycle_disp'] +
             intercept_value_max_stiff, 'r')
    plt.plot([], ' ', label=f'stiffness = {stiffness:.0f} N/mm')
    plt.ylabel('force lc / N')
    plt.xlabel('disp ext / mm')
    plt.legend()
    savepath = Cwd / '04_Results/00_Mineralized/00_force_disp/'
    plt.savefig(os.path.join(savepath, 'force_disp_' + sample_ID + '.png'), dpi=300)
    plt.show()
    # plt.close()

    # calculate stress/strain
    Pi = 3.14159265
    l_initial = 6.5
    area = results_uCT['min_Area'][counter]
    stress = df['force_lc'] / area
    strain = df['disp_ext'] / l_initial
    df['stress_lc'] = stress
    df['strain_ext'] = strain
    df['stress_lc_filtered'] = butter_lowpass_filter(df['stress_lc'], cutoff)
    counter = counter + 1

    # plot stress/strain
    plt.figure()
    plt.title(sample_ID)
    plt.plot(df['strain_ext'], df['stress_lc'], label='raw')
    plt.plot(df['strain_ext'], df['stress_lc_filtered'], label='filtered')
    plt.ylabel('stress / MPa')
    plt.xlabel('strain / -')
    plt.legend()
    savepath = Cwd / '04_Results/00_Mineralized/01_stress_strain/'
    plt.savefig(os.path.join(savepath, 'stress_strain_' + sample_ID + '.png'), dpi=300)
    plt.close()

    # calculate values for last cycle of stress/strain curve only
    last_cycle_stress = df['stress_lc_filtered'][peaks_index[-1]:]
    last_cycle_strain = df['strain_ext'][peaks_index[-1]:]
    last_cycle_strain = last_cycle_strain.dropna().reset_index(drop=True)
    last_cycle_stress = last_cycle_stress.dropna().reset_index(drop=True)

    # plot last cycle of stress/strain curve
    plt.figure()
    plt.title(sample_ID)
    plt.plot(last_cycle_strain, last_cycle_stress)
    plt.ylabel('stress / MPa')
    plt.xlabel('strain / -')
    # plt.show()
    plt.close()

    # calculate apparent modulus of elasticity using regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(last_cycle_strain, last_cycle_stress)
    youngs_modulus = round(slope,1)

    ## calculate apparent modulus by using rolling regression
    # definition of strain region where regression should be carried out
    upper_strain = 0.00250
    lower_strain = 0.00100

    # isolate stress/strain values of defined strain region
    last_cycle = pd.DataFrame()
    last_cycle['last_cycle_strain'] = round(last_cycle_strain, 5)
    last_cycle['last_cycle_stress'] = round(last_cycle_stress, 5)

    # identify index range of defined region
    upper_cond = last_cycle.loc[last_cycle['last_cycle_strain'] == upper_strain]
    lower_cond = last_cycle.loc[last_cycle['last_cycle_strain'] == lower_strain]
    max_strain_ind = min(upper_cond.index)
    min_strain_ind = min(lower_cond.index)
    last_cycle = last_cycle[max_strain_ind:min_strain_ind]

    slope_values_app = list()
    intercept_values_app = list()

    # rolling linear regression for apparent modulus calculation
    for x in range(max_strain_ind, min_strain_ind - window_width + 1, 1):
        last_cycle_mod_app = last_cycle[x:x+window_width]
        slope_app, intercept_app, r_value, p_value, std_err = stats.linregress(last_cycle_mod_app['last_cycle_strain'],
                                                                       last_cycle_mod_app['last_cycle_stress'])
        slope_value_app = slope_app
        slope_values_app.append(slope_value_app)
        intercept_value_app = intercept_app
        intercept_values_app.append(intercept_value_app)
    slope_value_app_df = pd.DataFrame(slope_values_app)
    intercept_values_app_df = pd.DataFrame(intercept_values_app)
    apparent_modulus = slope_value_app_df.max()[0]
    max_slope_index_app = slope_value_app_df[slope_value_app_df == apparent_modulus].dropna()
    max_slope_index_app = max_slope_index_app.index
    intercept_value_max_app = intercept_values_app_df.loc[max_slope_index_app[0]].values[0]
    last_cycle_plot = last_cycle[max_slope_index_app[0]:max_slope_index_app[0] + window_width]

    # create list with current values which are sample_ID, slope & apparent modulus & add them to result list which
    # is then converted to dataframe
    values = [sample_ID, round(stiffness), round(apparent_modulus)]
    result.append(values)
    result_dir = pd.DataFrame(result, columns=['Sample ID', 'Stiffness N/mm', 'Apparent modulus MPa'])

    # generate plot
    plt.figure(figsize=(6, 4))
    plt.title(sample_ID)
    plt.plot(df['strain_ext'], df['stress_lc_filtered'], label='filtered')
    plt.plot(last_cycle['last_cycle_strain'], last_cycle['last_cycle_stress'], label='last unloading cycle 0.001-0.0025')
    plt.plot(last_cycle_plot['last_cycle_strain'], last_cycle_plot['last_cycle_stress'], label='regress area',
             color='k')
    plt.plot(last_cycle_plot['last_cycle_strain'], apparent_modulus * last_cycle_plot['last_cycle_strain'] +
             intercept_value_max_app, 'r')
    plt.plot([], ' ', label=f'apparent modulus = {apparent_modulus:.0f} MPa')
    plt.ylabel('stress / MPa')
    plt.xlabel('strain / -')
    plt.legend()
    savepath = Cwd / '04_Results/00_Mineralized/01_stress_strain/'
    plt.savefig(os.path.join(savepath, 'stress_strain_' + sample_ID + '.png'), dpi=300)
    plt.show()
    # plt.close()


missing_sample_IDs = pd.DataFrame({'Sample ID': ['390R', '395R', '400R', '402L', '433L']})
result_dir = pd.concat([result_dir, missing_sample_IDs])
result_dir_sorted = result_dir.sort_values(by=['Sample ID'], ascending=True)

print(result_dir_sorted)

# safe dataframe to csv
result_dir_sorted.to_csv(
    os.path.join('/home/stefan/Documents/PythonScripts/04_Results/00_Mineralized/', 'ResultsElasticTesting.csv'),
    index=False)

