# This script loads data obtained during experimental tensile testing on the MTS. Force/displacement data is filtered
# and used to calculate the corresponding stress/strain values. Stiffness/apparent modulus were extracted from the
# respective slopes. The measures were calculated as follows:
# Apparent modulus: stress/strain; stress = filtered force/mean apparent area
# Stiffness: filtered force/displacement

import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from scipy import stats
from pathlib import Path
from matplotlib import rcParams

# definition of lowpass filter
def butter_lp_filt(data, cutoff, order=5):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


# definition of path
Cwd = Path.cwd()
DataPath = Cwd / 'Silk/00_Silk_data/01_mechanical_testing/'
filename_list = [File for File in os.listdir(DataPath) if File.endswith('.csv')]
filename_list.sort()

# load uCT results & remove naN entries; areas needed for stress calculations
results_uCT = pd.read_csv(str('/home/stefan/Documents/PythonScripts/Silk/02_Silk_results/00_uCT/CrossSectionalArea.csv')
                          , skiprows=0)
# results_uCT = pd.read_csv(str('C:/Users/Stefan/PycharmProjects/FEMCOL/04_Results/03_uCT/ResultsUCT.csv'), skiprows=0)
results_uCT = results_uCT.drop(index=[0], axis=0)
results_uCT = results_uCT.reset_index(drop=True)

# set counters for iterations over files (i) and areas for stress calculation (counter) & initialize results list
result = list()
i = 0

# loop over .csv files in Folder
for filename in filename_list:
    # sample_ID = filename.split('/')[-1].split('_')[0]
    sample_ID = filename.split('/')[-1].split('.')[0]
    # load csv:
    df = pd.read_csv(str(DataPath / filename_list[i]), skiprows=2)
    df.rename(columns={'sec': 'time', 'N': 'force', 'mm': 'disp'}, inplace=True)
    i = i + 1

    # filter signals:
    fs = 102.4  # sample rate, Hz
    cutoff = 10
    nyq = 0.5 * fs
    df['force_filt'] = butter_lp_filt(df['force'], cutoff)

    # plot filtered signals (displacement extensometer vs. force MTS)
    plt.figure()
    plt.title(sample_ID)
    plt.plot(df['disp'], df['force'], label='raw')
    plt.plot(df['disp'], df['force_filt'], label='filtered')
    plt.ylabel('force / N')
    plt.xlabel('disp / mm')
    plt.legend()
    # plt.show()
    plt.close()

    # peak detection
    peaks_index, _ = find_peaks(df['force'], width=200, prominence=0.7)
    plt.figure()
    plt.title(sample_ID)
    plt.plot(df['disp'], df['force'], label='raw')
    plt.plot(df['disp'], df['force_filt'], label='filtered')
    plt.scatter(df['disp'][peaks_index], df['force'][peaks_index], marker='o')
    plt.ylabel('force / N')
    plt.xlabel('disp / mm')
    plt.legend()
    # plt.show()
    plt.close()

    preconditioning = df[(df.force_filt >= 9) & (df.force_filt <= 41) & (df.disp <= 15)]
    preconditioning.reset_index(drop=True, inplace=True)

    plt.figure()
    plt.title(sample_ID)
    plt.plot(preconditioning['disp'], preconditioning['force'])
    # plt.scatter(df['disp'][peaks_index], df['force'][peaks_index], marker='o', color='black')
    plt.scatter(df['disp'][peaks_index[0:13]], df['force_filt'][peaks_index[0:13]], marker='o', color='red')
    plt.ylabel('force / N')
    plt.xlabel('disp / mm')
    plt.show()

    precond_unload = df[(df.cycle == 14)]
    window_width = round(1/3 * len(precond_unload))

    # rolling linear regression for stiffness calculation
    slope_values_stiff = list()
    intercept_values_stiff = list()
    for x in range(0, len(precond_unload) - 1 - window_width + 1, 1):
        precond_unload_mod = precond_unload[x:x + window_width]
        slope_stiff, intercept_stiff, r_value, p_value, std_err = stats.linregress(precond_unload_mod['disp'],
                                                                                   precond_unload_mod['force'])
        # collect slope value & add to growing list; same for intercept
        slope_value_stiff = slope_stiff
        intercept_value = intercept_stiff
        slope_values_stiff.append(slope_value_stiff)
        intercept_values_stiff.append(intercept_value)

    # Create DataFrames for slope or stiffness to search for max. values & indices; create cycle for plotting
    slope_value_df = pd.DataFrame(slope_values_stiff)
    intercept_values_df = pd.DataFrame(intercept_values_stiff)
    stiffness = slope_value_df.max()[0]
    max_slope_index = slope_value_df[slope_value_df == stiffness].dropna()
    max_slope_index = max_slope_index.index
    intercept_value_max_stiff = intercept_values_df.loc[max_slope_index[0]].values[0]
    last_cycle_plot = precond_unload[max_slope_index[0]:max_slope_index[0] + window_width]

    # generate plot
    # plt.figure(figsize=(6, 4))
    # plt.title(sample_ID)
    plt.plot(precond_unload['disp'], precond_unload['force'], label='Last unloading cycle')
    plt.plot(last_cycle_plot['disp'], stiffness * last_cycle_plot['disp'] +
             intercept_value_max_stiff, label='Fit', color='black')
    plt.plot([], ' ', label=f'Stiffness = {stiffness:.0f} N/mm')
    plt.plot([], ' ', label='Sample ID: ' + sample_ID)
    plt.ylabel('force / N')
    plt.xlabel('disp / mm')
    plt.legend()
    plt.autoscale()
    plt.rcParams.update({'font.size': 14})
    # plt.legend(prop={'size': 14})
    savepath_fd = Cwd / '04_Results/00_Mineralized/00_force_disp/'
    plt.savefig(os.path.join(savepath_fd, 'force_disp_el_' + sample_ID + '.png'), dpi=300, bbox_inches='tight', format='png')
    # plt.savefig(os.path.join(savepath_fd, 'force_disp_el_' + sample_ID + '.eps'), dpi=300, bbox_inches='tight', format='eps')
    plt.show()
    # plt.close()

    # calculate stress/strain, and put into dataframe
    l_initial = 50
    area = results_uCT['Mean Area'][0]
    stress = df['force_filt'] / area
    strain = df['disp'] / l_initial
    df['stress'] = stress
    df['strain'] = strain

    # plot stress/strain
    plt.figure()
    plt.title(sample_ID)
    plt.plot(df['strain_ext'], df['stress'], label='raw')
    plt.plot(df['strain_ext'], df['stress'], label='filtered')
    plt.ylabel('stress / MPa')
    plt.xlabel('strain / -')
    plt.legend()
    anpassen savepath = Cwd / '04_Results/00_Mineralized/01_stress_strain/'
    plt.savefig(os.path.join(savepath, 'stress_strain_' + sample_ID + '.png'), dpi=300, format='png')
    # plt.savefig(os.path.join(savepath, 'stress_strain_' + sample_ID + '.eps'), dpi=300, format='eps')
    plt.close()

    # calculate values for last cycle of stress/strain curve only
    stress_lc = precond_unload['force_filt'] / area
    strain_lc = precond_unload['disp'] / l_initial
    precond_unload['stress_lc'] = stress_lc
    precond_unload['strain_lc'] = strain_lc

    # plot last cycle of stress/strain curve
    plt.figure()
    plt.title(sample_ID)
    plt.plot(precond_unload['strain_lc'], precond_unload['stress_lc'])
    plt.ylabel('stress / MPa')
    plt.xlabel('strain / -')
    # plt.show()
    plt.close()

    ## calculate apparent modulus by using rolling regression
    # initialize lists for slope/intercept value collection
    slope_values_app = list()
    intercept_values_app = list()

    # rolling linear regression for apparent modulus calculation
    for x in range(precond_unload['disp_lc'].max(), precond_unload['disp_lc'].min() - window_width + 1, 1):
        last_cycle_mod = precond_unload[x:x+window_width]
        slope_app, intercept_app, r_value, p_value, std_err = stats.linregress(last_cycle_mod['strain_lc'],
                                                                               last_cycle_mod['stress_lc'])
        # collect slope value & add to growing list; same for intercept
        slope_value_app = slope_app
        slope_values_app.append(slope_value_app)
        intercept_value_app = intercept_app
        intercept_values_app.append(intercept_value_app)

    # Create DataFrames for slope or stiffness & intercept to search for max. values & indices; create cycle for plotting
    slope_value_app_df = pd.DataFrame(slope_values_app)
    intercept_values_app_df = pd.DataFrame(intercept_values_app)
    apparent_modulus = slope_value_app_df.max()[0]
    max_slope_index_app = slope_value_app_df[slope_value_app_df == apparent_modulus].dropna()
    max_slope_index_app = max_slope_index_app.index
    intercept_value_max_app = intercept_values_app_df.loc[max_slope_index_app[0]].values[0]
    last_cycle_plot = precond_unload[max_slope_index_app[0]:max_slope_index_app[0] + window_width]

    # generate plot
    # plt.figure(figsize=(6, 4))
    # plt.title(sample_ID)
    plt.plot(precond_unload['strain'], precond_unload['stress_lc'], label='last unloading cycle')
    # plt.plot(last_cycle_plot['last_cycle_strain'], last_cycle_plot['last_cycle_stress'], label='regress area', color='k')
    plt.plot(last_cycle_plot['strain_lc'], apparent_modulus * last_cycle_plot['strain_lc'] +
             intercept_value_max_app, label='Fit', color='black')
    plt.plot([], ' ', label=f'Apparent modulus = {apparent_modulus:.0f} MPa')
    plt.plot([], ' ', label='Sample ID: ' + sample_ID)
    plt.ylabel('stress / MPa')
    plt.xlabel('strain / -')
    plt.legend()
    plt.autoscale()
    plt.rcParams.update({'font.size': 14})
    # plt.legend(prop={'size': 14})
    anpassen savepath = Cwd / '04_Results/00_Mineralized/01_stress_strain/'
    plt.savefig(os.path.join(savepath, 'stress_strain_el_' + sample_ID + '.png'), dpi=300, bbox_inches='tight', format='png')
    # plt.savefig(os.path.join(savepath, 'stress_strain_el_' + sample_ID + '.eps'), dpi=300, bbox_inches='tight', format='eps')
    plt.show()
    # plt.close()

    # create list with current values which are sample_ID, slope & apparent modulus & add them to result list which
    # is then converted to dataframe
    values = [sample_ID, round(stiffness), round(apparent_modulus)]
    result.append(values)
    result_dir = pd.DataFrame(result, columns=['Sample ID', 'Stiffness N/mm', 'Apparent modulus MPa'])

    # rcParams.update({'figure.autolayout': True})
    # time = pd.DataFrame()
    # time = df['time'] - df['time'].loc[0]
    # fig, ax1 = plt.subplots()
    # ax1.plot(time, df['disp_ext'], label='Displacement')
    # ax2 = ax1.twinx()
    # ax2.plot(time, df['force_lc'], color='darkorange', label='Force')
    # plt.plot([], ' ', label='Sample ID: ' + sample_ID)
    # # plt.title(sample_ID)
    # ax1.set_xlabel('Time s')
    # ax1.set_ylabel('Displacement mm')
    # ax2.set_ylabel('Force N')
    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=3)
    # ax1.autoscale()
    # ax2.autoscale()
    # plt.rcParams.update({'font.size': 14})
    # savepath_new = 'C:/Users/Stefan/PycharmProjects/FEMCOL/04_Results/00_Mineralized/02_disp_force_time'
    # plt.savefig(os.path.join(savepath_new, 'disp_time_el_' + sample_ID + '.eps'), dpi=300, bbox_inches='tight', format='eps')
    # plt.show()

anpassen result_dir.to_csv(os.path.join('/home/stefan/Documents/PythonScripts/Silk/00_Mineralized/',
                                      'ResultsElasticTesting.csv'), index=False)
# result_dir_sorted.to_csv(os.path.join('C:/Users/Stefan/PycharmProjects/FEMCOL/04_Results/00_Mineralized',
#                                       'ResultsElasticTesting.csv'), index=False)




