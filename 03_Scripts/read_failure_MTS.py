# This script loads data obtained during experimental tensile testing on the MTS. Force/displacement data is filtered
# and used to calculate the corresponding stress/strain values. Ultimate values and stiffness/apparent modulus were
# extracted from the respective slopes. The measures were calculated as follows:
# Ultimate stress: filtered force/mean apparent area (mean apparent area over sample gauge length, extracted from uCT
# image)
# Apparent modulus: stress/strain; stress = filtered force/mean apparent area (same as for US but mean instead of min)
# Stiffness: filtered force/displacement

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from scipy import stats
from pathlib import Path
from matplotlib import rcParams
from tqdm import tqdm
import matplotlib.ticker as ticker

# definition of lowpass filter
def butter_lowpass_filter(data, cutoff, order=9):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


# definition of path
Cwd = os.getcwd()
DataPath = str(os.path.dirname(Cwd)) + '/02_Data/02_MTS/Failure_testing_demineralized/'
savepath_fd = str(os.path.dirname(Cwd)) + '/04_Results/01_Demineralized/00_force_disp/'
savepath_ss = str(os.path.dirname(Cwd)) + '/04_Results/01_Demineralized/01_stress_strain/'
savepath_dft = str(os.path.dirname(Cwd)) + '/04_Results/01_Demineralized/02_disp_force_time/'

filename_list = [File for File in os.listdir(DataPath) if File.endswith('.csv')]
filename_list.sort()

# load uCT results & remove naN entries; areas needed for stress calculations
results_uCT = pd.read_csv(str(os.path.dirname(Cwd) + '/04_Results/03_uCT/ResultsUCT.csv'), skiprows=0)

test = results_uCT.drop(results_uCT.loc[results_uCT['Sample ID'] == '390_R'].index)
test1 = test.drop(results_uCT.loc[results_uCT['Sample ID'] == '395_R'].index)
test2 = test1.drop(results_uCT.loc[results_uCT['Sample ID'] == '400_R'].index)
test3 = test2.drop(results_uCT.loc[results_uCT['Sample ID'] == '402_L'].index)
test4 = test3.drop(results_uCT.loc[results_uCT['Sample ID'] == '410_L'].index)
test5 = test4.drop(results_uCT.loc[results_uCT['Sample ID'] == '433_L'].index)
test6 = test5.drop(results_uCT.loc[results_uCT['Sample ID'] == '396_R'].index)
test7 = test6.drop(results_uCT.loc[results_uCT['Sample ID'] == '410_R'].index)
test8 = test7.drop(results_uCT.loc[results_uCT['Sample ID'] == '403_R'].index)

results_uCT = test8.reset_index(drop=True)

if len(results_uCT) != len(filename_list):
    print('+++++++ Length of results_uCT has to be equal to length of filename_list in order to calculate stress '
          'correctly +++++++')
    exit()
else:
    # set counters for iterations over files (i) and areas for stress calculation (counter) & initialize results list
    i = 0
    counter = 0
    result = list()

    # loop over .csv files in Folder
    for filename in tqdm(filename_list):
        sample_ID = filename.split('/')[-1].split('_')[0]
        # load csv:
        df = pd.read_csv(str(DataPath + filename_list[i]), skiprows=2)
        df.rename(columns={'sec': 'time', 'N': 'force_MTS', 'N.1': 'force_lc', 'mm': 'disp_MTS', 'mm.1': 'disp_ext'},
                  inplace=True)
        i = i + 1
        # print(sample_ID)
        # filter signals

        fs = 102.4  # sample rate, Hz
        cutoff = 5
        nyq = 0.5 * fs
        df['force_lc_filtered'] = butter_lowpass_filter(df['force_lc'], cutoff)

        # # +++++++++++++++  this part was used for end of curve analysis; corrections were done in the raw files by
        # # removal of last columns

        # filename = '431L_fail.csv'
        # sample_ID = '431L'
        # df = pd.read_csv(str(DataPath / filename), skiprows=2)
        # df.rename(columns={'sec': 'time', 'N': 'force_MTS', 'N.1': 'force_lc', 'mm': 'disp_MTS', 'mm.1': 'disp_ext'},
        #           inplace=True)
        # fs = 102.4  # sample rate, Hz
        # cutoff = 5
        # nyq = 0.5 * fs
        # df['force_lc_filtered'] = butter_lowpass_filter(df['force_lc'], cutoff)
        # peaks_index, _ = find_peaks(df['force_lc'], width=200)
        #
        # plt.plot(df['disp_ext'], df['force_lc_filtered'], label='Filtered')
        # plt.plot(df['disp_ext'], df['force_lc'], label='raw')
        # plt.scatter(df['disp_ext'][peaks_index[-1]], df['force_lc_filtered'][peaks_index[-1]], label='peak')
        # plt.ylabel('force / N')
        # plt.xlabel('disp / mm')
        # plt.legend()
        # plt.show()

        # # +++++++++++++++++

        # calculate stress and strain, filter & put into dataframe
        l_initial = 6.5
        min_area_wp = results_uCT['Min ECM Area / mm^2'][counter]       # needed to calculate ultimate collagen stress
        mean_bone_area_wp = results_uCT['Mean ECM Area / mm^2'][counter]
        mean_area_wop = results_uCT['Mean Apparent Area / mm^2'][counter]
        collagen_stress = df['force_lc'] / min_area_wp
        stress_bone_wp = df['force_lc'] / mean_bone_area_wp     # needed to calculate elastic modulus/ultimate stress
        stress_wop = df['force_lc'] / mean_area_wop     # needed to calculate apparent modulus/ultimate stress
        strain = df['disp_ext'] / l_initial
        df['collagen_stress'] = collagen_stress
        df['stress_lc_wop'] = stress_wop
        df['stress_bone_wp'] = stress_bone_wp
        df['strain_ext'] = strain
        df['collagen_stress_filtered'] = butter_lowpass_filter(df['collagen_stress'], cutoff)
        df['stress_lc_filtered_wop'] = butter_lowpass_filter(df['stress_lc_wop'], cutoff)
        df['stress_bone_wp_filtered'] = butter_lowpass_filter(df['stress_bone_wp'], cutoff)
        counter = counter + 1

        # find max value of stress_lc_wop column & locate index
        column = df['stress_lc_wop']
        max_stress_raw = column.max()
        row_max_stress = df.loc[df['stress_lc_wop'] == max_stress_raw]
        max_stress_index = row_max_stress.index

        # find max value of collagen_stress column & locate index
        column_cs = df['collagen_stress']
        max_collagen_stress_raw = column_cs.max()
        row_max_collagen_stress = df.loc[df['collagen_stress'] == max_collagen_stress_raw]
        max_collagen_stress_index = row_max_collagen_stress.index

        # find max value of stress_bone_wp column & locate index
        column_b = df['stress_bone_wp']
        max_stress_b_raw = column_b.max()
        row_max_stress_b = df.loc[df['stress_bone_wp'] == max_stress_b_raw]
        max_stress_b_index = row_max_stress_b.index

        # search for apparent ultimate stress (filtered)
        row_max_stress_filtered = df.iloc[max_stress_index]
        ultimate_stress_filtered = np.round(row_max_stress_filtered['stress_lc_filtered_wop'].values, 2)
        ultimate_stress_filtered = ultimate_stress_filtered[0]

        # search for ultimate collagen stress (filtered)
        row_max_collagen_stress_filtered = df.iloc[max_collagen_stress_index]
        ultimate_collagen_stress_filtered = np.round(row_max_collagen_stress_filtered['collagen_stress_filtered'].values, 2)
        ultimate_collagen_stress_filtered = ultimate_collagen_stress_filtered[0]

        # search for ultimate stress (filtered)
        row_max_stress_b_filtered = df.iloc[max_stress_b_index]
        ultimate_stress_b_filtered = np.round(row_max_stress_b_filtered['stress_bone_wp_filtered'].values, 2)
        ultimate_stress_b_filtered = ultimate_stress_b_filtered[0]

        # search for ultimate strain
        row_max_strain = df.iloc[max_stress_index]
        ultimate_strain = np.round(row_max_strain['strain_ext'].values, 4)
        ultimate_strain = ultimate_strain[0]

        # find max value of force_lc column & find ultimate force (raw)
        column_force = df['force_lc']
        max_force_raw = column_force.max()
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
        start_range_strain = df['strain_ext'][peaks_index[6]:peaks_index[7]]
        start_range_stress = df['stress_lc_filtered_wop'][peaks_index[6]:peaks_index[7]]
        start_range = pd.DataFrame()
        start_range['strain_ext'] = start_range_strain
        start_range['stress_lc_filtered_wop'] = start_range_stress

        # find smallest stress value & corresponding index as starting point
        start_value = min(start_range['stress_lc_filtered_wop'])
        start_ind = start_range.loc[start_range['stress_lc_filtered_wop'] == start_value]
        start_index = start_ind.index

        # create dataframe using unloading part of last cycle
        last_cycle = pd.DataFrame()
        last_cycle['last_cycle_strain'] = df['strain_ext'].iloc[start_index[0]:peaks_index[7]]
        last_cycle['last_cycle_stress'] = df['stress_lc_filtered_wop'].iloc[start_index[0]:peaks_index[7]]

        ## calculate stiffness using last unloading cycle of force/displacement data
        # create dataframe of final unloading cycle using start/end index (same as for stress/strain)
        last_cycle_force = df['force_lc_filtered'].iloc[start_index[0]:peaks_index[7]]
        last_cycle_disp = df['disp_ext'].iloc[start_index[0]:peaks_index[7]]
        last_cycle_disp = last_cycle_disp.dropna().reset_index(drop=True)
        last_cycle_force = last_cycle_force.dropna().reset_index(drop=True)

        last_cycle_fd = pd.DataFrame()
        last_cycle_fd['last_cycle_disp'] = round(last_cycle_disp, 5)
        last_cycle_fd['last_cycle_force'] = round(last_cycle_force, 5)

        # define window width & initialize lists for slope/intercept value collection
        window_width = round(1 / 3 * len(last_cycle_fd))
        slope_values_stiff = list()
        intercept_values_stiff = list()

        # rolling linear regression
        for k in range(0, len(last_cycle_fd) - 1 - window_width + 1, 1):
            last_cycle_mod = last_cycle_fd[k:k + window_width]
            slope_stiff, intercept_stiff, r_value, p_value, std_err = stats.linregress(last_cycle_mod['last_cycle_disp'],
                                                                                       last_cycle_mod['last_cycle_force'])

            # collect slope value & add to growing list; same for intercept
            slope_value_stiff = slope_stiff
            slope_values_stiff.append(slope_value_stiff)
            intercept_value_stiff = intercept_stiff
            intercept_values_stiff.append(intercept_value_stiff)

        # Create DataFrames for slope or stiffness & intercept to search for max. values & indices; create cycle for plotting
        slope_value_stiff_df = pd.DataFrame(slope_values_stiff)
        intercept_values_stiff_df = pd.DataFrame(intercept_values_stiff)
        stiffness = slope_value_stiff_df.max()[0]
        max_slope_index_stiff = slope_value_stiff_df[slope_value_stiff_df == stiffness].dropna()
        max_slope_index_stiff = max_slope_index_stiff.index
        intercept_value_max_stiff = intercept_values_stiff_df.loc[max_slope_index_stiff[0]].values[0]
        last_cycle_plot_fd = last_cycle_fd[max_slope_index_stiff[0]:max_slope_index_stiff[0] + window_width]

        # generate plot
        # plt.figure(figsize=(6, 4))
        # plt.title(sample_ID)
        plt.plot(df['disp_ext'][0:peaks_index[-1]], df['force_lc_filtered'][0:peaks_index[-1]], label='Filtered')
        plt.plot(last_cycle_fd['last_cycle_disp'], last_cycle_fd['last_cycle_force'], label='Lasting unloading cycle')
        # plt.plot(last_cycle_plot_fd['last_cycle_disp'], last_cycle_plot_fd['last_cycle_force'], label='regress area',
        #          color='k')
        plt.plot(last_cycle_plot_fd['last_cycle_disp'], stiffness * last_cycle_plot_fd['last_cycle_disp'] +
                 intercept_value_max_stiff, label='Fit', color='black')
        plt.plot([], ' ', label=f'Stiffness = {stiffness:.0f} N/mm')
        plt.plot([], ' ', label='Sample ID: ' + sample_ID)
        plt.ylabel('force / N')
        plt.xlabel('disp / mm')
        plt.legend()
        plt.autoscale()
        plt.rcParams.update({'font.size': 14})
        plt.savefig(os.path.join(savepath_fd, 'force_disp_fail_' + sample_ID + '.png'), dpi=300, bbox_inches='tight', format='png')
        # plt.show()
        plt.close()

        ## calculate apparent modulus using last unloading cycle
        # create dataframe of final unloading cycle using start/end index (same as above)
        last_cycle_stress = df['stress_lc_filtered_wop'].iloc[start_index[0]:peaks_index[7]]
        last_cycle_strain = df['strain_ext'].iloc[start_index[0]:peaks_index[7]]
        last_cycle_strain = last_cycle_strain.dropna().reset_index(drop=True)
        last_cycle_stress = last_cycle_stress.dropna().reset_index(drop=True)

        last_cycle_stress_b = df['stress_bone_wp_filtered'].iloc[start_index[0]:peaks_index[7]]
        last_cycle_strain_b = df['strain_ext'].iloc[start_index[0]:peaks_index[7]]

        last_cycle_ss = pd.DataFrame()
        last_cycle_ss['last_cycle_strain'] = round(last_cycle_strain, 5)
        last_cycle_ss['last_cycle_stress'] = round(last_cycle_stress, 5)

        last_cycle_sb = pd.DataFrame()
        last_cycle_sb['last_cycle_strain'] = round(last_cycle_strain_b, 5)
        last_cycle_sb['last_cycle_stress'] = round(last_cycle_stress_b, 5)

        # define window width (same as for force/disp) & initialize lists for slope/intercept value collection
        window_width = round(1 / 3 * len(last_cycle_ss))
        slope_values_app = list()
        intercept_values_app = list()
        slope_values_b = list()
        intercept_values_b = list()

        # rolling linear regression to calculate apparent modulus of last unloading cycle
        for k in range(len(last_cycle_ss) - 1 - window_width + 1):
            last_cycle_mod = last_cycle_ss[k:k + window_width]
            last_cycle_mod_b = last_cycle_sb[k:k + window_width]
            slope_app, intercept_app, r_value, p_value, std_err = stats.linregress(last_cycle_mod['last_cycle_strain'],
                                                                                   last_cycle_mod['last_cycle_stress'])
            slope_b, intercept_b, r_value_b, p_value_b, std_err_b = stats.linregress(last_cycle_mod_b['last_cycle_strain'],
                                                                                     last_cycle_mod_b['last_cycle_stress'])

            # collect slope/intercept values & add to growing lists
            slope_value_app = slope_app
            slope_values_app.append(slope_value_app)
            intercept_value_app = intercept_app
            intercept_values_app.append(intercept_value_app)

            slope_value_b = slope_b
            slope_values_b.append(slope_value_b)
            intercept_value_b = intercept_b
            intercept_values_b.append(intercept_value_b)

        # Create DataFrames for slope/app. modulus to search for max. values & corresponding indices; create cycle for plot
        slope_value_app_df = pd.DataFrame(slope_values_app)
        intercept_values_app_df = pd.DataFrame(intercept_values_app)
        apparent_modulus = slope_value_app_df.max()[0]
        max_slope_index_app = slope_value_app_df[slope_value_app_df == apparent_modulus].dropna()
        max_slope_index_app = max_slope_index_app.index
        intercept_value_max_app = intercept_values_app_df.loc[max_slope_index_app[0]].values[0]
        last_cycle_plot_ss = last_cycle_ss[max_slope_index_app[0]:max_slope_index_app[0] + window_width]

        slope_value_b_df = pd.DataFrame(slope_values_b)
        intercept_values_b_df = pd.DataFrame(intercept_values_b)
        modulus_b = slope_value_b_df.max()[0]
        max_slope_index_b = slope_value_b_df[slope_value_b_df == modulus_b].dropna()
        max_slope_index_b = max_slope_index_b.index
        intercept_value_max_b = intercept_values_b_df.loc[max_slope_index_b[0]].values[0]
        last_cycle_plot_sb = last_cycle_sb[max_slope_index_b[0]:max_slope_index_b[0] + window_width]

        # generate plot
        # plt.figure(figsize=(6, 4))
        # plt.title(sample_ID)
        plt.plot(df['strain_ext'][0:peaks_index[-1]], df['stress_lc_filtered_wop'][0:peaks_index[-1]], label='Filtered')
        plt.plot(last_cycle_ss['last_cycle_strain'], last_cycle_ss['last_cycle_stress'], label='Last unloading cycle')
        # plt.plot(last_cycle_plot_ss['last_cycle_strain'], last_cycle_plot_ss['last_cycle_stress'], label='regress area',
        #          color='k')
        plt.plot(last_cycle_plot_ss['last_cycle_strain'], apparent_modulus * last_cycle_plot_ss['last_cycle_strain'] +
                 intercept_value_max_app, label='Fit', color='black')
        plt.plot([], ' ', label=f'Apparent modulus = {apparent_modulus:.0f} MPa')
        plt.plot([], ' ', label='Sample ID: ' + sample_ID)
        plt.ylabel('stress / MPa')
        plt.xlabel('strain / -')
        plt.legend()
        plt.autoscale()
        plt.rcParams.update({'font.size': 14})
        plt.savefig(os.path.join(savepath_ss, 'stress_strain_fail_' + sample_ID + '.png'), dpi=300, bbox_inches='tight', format='png')
        # plt.show()
        plt.close()

        # collect all data in list
        values = [sample_ID, ultimate_stress_filtered, ultimate_collagen_stress_filtered, ultimate_stress_b_filtered,
                  ultimate_strain, ultimate_force_filtered, round(apparent_modulus, 1), round(modulus_b, 1),
                  round(stiffness, 1)]

        # update list with each iteration's data & generate dataframe
        result.append(values)
        result_dir = pd.DataFrame(result, columns=['Sample ID', 'Ultimate stress / MPa', 'Ultimate collagen stress / MPa',
                                                   'Ultimate stress non-app / MPa', 'Ultimate strain / -',
                                                   'Ultimate Force / N', 'Apparent modulus / MPa',
                                                   'Modulus demineralized / MPa', 'Stiffness / N/mm'])

        rcParams.update({'figure.autolayout': True})
        time = pd.DataFrame()
        time = df['time'] - df['time'].loc[0]
        fig, ax1 = plt.subplots()
        ax1.plot(time[0:peaks_index[-1]], df['disp_ext'][0:peaks_index[-1]], label='Displacement')
        ax2 = ax1.twinx()
        ax2.plot(time[0:peaks_index[-1]], df['force_lc'][0:peaks_index[-1]], color='darkorange', label='Force')
        plt.plot([], ' ', label='Sample ID: ' + sample_ID)
        ax1.set_xlabel('Time s')
        ax1.set_ylabel('Displacement mm')
        ax2.set_ylabel('Force N')
        # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=2)
        fig.legend(loc='upper left', bbox_to_anchor=(0.14, 0.95), ncol=1)
        ax2.autoscale()
        plt.rcParams.update({'font.size': 14})
        plt.savefig(os.path.join(savepath_dft, 'disp_time_fail_' + sample_ID + '.png'), dpi=300, bbox_inches='tight', format='png')
        # plt.show()
        plt.close()

    # add missing samples to list
    # missing_sample_IDs = pd.DataFrame({'Sample ID': ['390R', '395R', '396R', '400R', '402L', '403R', '410L', '410R', '422R',
    #                                                  '431L', '433L']})
    missing_sample_IDs = pd.DataFrame({'Sample ID': ['390R', '395R', '396R', '400R', '402L', '403R', '410L', '410R',
                                                     '433L']})
    result_dir = pd.concat([result_dir, missing_sample_IDs])
    result_dir_sorted = result_dir.sort_values(by=['Sample ID'], ascending=True)
    #
    result_dir_sorted.to_csv(os.path.join(str(os.path.dirname(Cwd)) + '/04_Results/01_Demineralized/',
                                          'ResultsFailureTesting.csv'), index=False)

# generate force-disp plot for paper
df_393L_path = str(DataPath + '393L_fail.csv')
df_406L_path = str(DataPath + '406L_fail.csv')
df_426R_path = str(DataPath + '426R_fail.csv')

df_393L = pd.read_csv(str(df_393L_path), skiprows=2)
df_406L = pd.read_csv(str(df_406L_path), skiprows=2)
df_426R = pd.read_csv(str(df_426R_path), skiprows=2)
df_393L.rename(columns={'sec': 'time', 'N': 'force_MTS', 'N.1': 'force_lc', 'mm': 'disp_MTS', 'mm.1': 'disp_ext'},
                  inplace=True)
df_406L.rename(columns={'sec': 'time', 'N': 'force_MTS', 'N.1': 'force_lc', 'mm': 'disp_MTS', 'mm.1': 'disp_ext'},
                  inplace=True)
df_426R.rename(columns={'sec': 'time', 'N': 'force_MTS', 'N.1': 'force_lc', 'mm': 'disp_MTS', 'mm.1': 'disp_ext'},
                  inplace=True)

df_393L['force_lc_filt'] = butter_lowpass_filter(df_393L['force_lc'], cutoff)
df_406L['force_lc_filt'] = butter_lowpass_filter(df_406L['force_lc'], cutoff)
df_426R['force_lc_filt'] = butter_lowpass_filter(df_426R['force_lc'], cutoff)

# generate plot
# plt.figure(figsize=(6, 4))
# plt.title(sample_ID)
plt.plot(df_393L['disp_ext'][:30413], df_393L['force_lc'][:30413], label='393L')
plt.plot(df_406L['disp_ext'][:26765], df_406L['force_lc'][:26765], label='406L')
plt.plot(df_426R['disp_ext'][:41256], df_426R['force_lc'][:41256], label='426R')
# plt.plot([], ' ', label=f'Stiffness =  N/mm')
# plt.plot([], ' ', label='Sample ID: ' )
plt.ylabel('Force $F$ / N')
plt.xlabel('Displacement $\Delta l$ / mm')
plt.subplots_adjust(left=0.17, bottom=0.15, right=0.95, top=0.95)
plt.legend()
plt.rcParams['figure.figsize'] = (5.5, 4)
plt.rcParams.update({'font.size': 12})
ax = plt.gca()
ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
plt.savefig(os.path.join(savepath_fd, 'force_disp_paper.jpg'), dpi=1200, format='jpg', pad_inches=0)
plt.show()
# plt.close()
