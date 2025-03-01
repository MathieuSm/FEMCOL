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
from matplotlib import rcParams
from tqdm import tqdm

# definition of lowpass filter
def butter_lowpass_filter(data, cutoff, order=9):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


# definition of path
Cwd = os.getcwd()
DataPath = str(os.path.dirname(Cwd)) + '/02_Data/02_MTS/Elastic_testing_mineralized/'
savepath_fd = str(os.path.dirname(Cwd)) + '/04_Results/00_Mineralized/00_force_disp/'
savepath_ss = str(os.path.dirname(Cwd)) + '/04_Results/00_Mineralized/01_stress_strain/'
savepath_dft = str(os.path.dirname(Cwd)) + '/04_Results/00_Mineralized/02_disp_force_time'

filename_list = [File for File in os.listdir(DataPath) if File.endswith('.csv')]
filename_list.sort()

# load uCT results & remove naN entries; areas needed for stress calculations
results_uCT = pd.read_csv(str(os.path.dirname(Cwd) + '/04_Results/03_uCT/ResultsUCT.csv'), skiprows=0)

test = results_uCT.drop(results_uCT.loc[results_uCT['Sample ID'] == '390_R'].index)
test1 = test.drop(results_uCT.loc[results_uCT['Sample ID'] == '395_R'].index)
test2 = test1.drop(results_uCT.loc[results_uCT['Sample ID'] == '400_R'].index)
test3 = test2.drop(results_uCT.loc[results_uCT['Sample ID'] == '402_L'].index)
test4 = test3.drop(results_uCT.loc[results_uCT['Sample ID'] == '410_L'].index)
test5 = test4.drop(results_uCT.loc[results_uCT['Sample ID'] == '433_R'].index)
results_uCT = test5.reset_index(drop=True)

# set counters for iterations over files (i) and areas for stress calculation (counter) & initialize results list
result = list()
i = 0
counter = 0
# # +++++++++++++++ this part was used for end of curve analysis +++++++++++++++
# filename = '386R_el.csv'
# sample_ID = '386R'
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

# loop over .csv files in Folder
for filename in tqdm(filename_list):
    sample_ID = filename.split('/')[-1].split('_')[0]
    # load csv:
    df = pd.read_csv(str(DataPath + filename_list[i]), skiprows=2)
    df.rename(columns={'sec': 'time', 'N': 'force_MTS', 'N.1': 'force_lc', 'mm': 'disp_MTS', 'mm.1': 'disp_ext'},
              inplace=True)
    i = i + 1

    # filter signals:
    fs = 102.4  # sample rate, Hz
    cutoff = 5
    nyq = 0.5 * fs
    df['force_lc_filtered'] = butter_lowpass_filter(df['force_lc'], cutoff)

    # plot filtered signals (displacement extensometer vs. force MTS)
    # plt.figure()
    # plt.title(sample_ID)
    # plt.plot(df['disp_ext'], df['force_lc'], label='raw')
    # plt.plot(df['disp_ext'], df['force_lc_filtered'], label='filtered')
    # plt.ylabel('force lc / N')
    # plt.xlabel('disp ext / mm')
    # plt.legend()
    # plt.show()
    # plt.close()

    # peak detection
    peaks_index, _ = find_peaks(df['force_lc'], width=50)

    # calculate values for last cycle of force/disp curve only
    last_cycle_force = df['force_lc_filtered'][peaks_index[-1]:]
    last_cycle_disp = df['disp_ext'][peaks_index[-1]:]
    last_cycle_disp = last_cycle_disp.dropna().reset_index(drop=True)
    last_cycle_force = last_cycle_force.dropna().reset_index(drop=True)

    # isolate force/displacement values
    last_cycle = pd.DataFrame()
    last_cycle['last_cycle_force'] = round(last_cycle_force, 5)
    last_cycle['last_cycle_disp'] = round(last_cycle_disp, 5)

    # set boundaries for further calculation (represent approx. linear region)
    upper_disp = max(last_cycle['last_cycle_disp'])
    lower_disp = 0.005

    # identify index range of defined region
    upper_cond_disp = last_cycle.loc[last_cycle['last_cycle_disp'] == upper_disp]
    lower_cond_disp = last_cycle.loc[last_cycle['last_cycle_disp'] >= lower_disp]
    max_disp_ind = min(upper_cond_disp.index)
    min_disp_ind = max(lower_cond_disp.index)
    last_cycle = last_cycle[max_disp_ind:min_disp_ind]

    # define window width for moving linear regression
    window_width = round(1 / 2 * len(last_cycle))

    # rolling linear regression for stiffness calculation
    slope_values_stiff = list()
    intercept_values_stiff = list()
    for x in range(0, len(last_cycle) - 1 - window_width + 1, 1):
        last_cycle_mod = last_cycle[x:x + window_width]
        slope_stiff, intercept_stiff, r_value, p_value, std_err = stats.linregress(last_cycle_mod['last_cycle_disp'],
                                                                                   last_cycle_mod['last_cycle_force'])
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
    last_cycle_plot = last_cycle[max_slope_index[0]:max_slope_index[0] + window_width]

    # generate plot
    # plt.figure(figsize=(6, 4))
    # plt.title(sample_ID)
    plt.plot(df['disp_ext'], df['force_lc_filtered'], label='Filtered')
    plt.plot(last_cycle['last_cycle_disp'], last_cycle['last_cycle_force'], label='Last unloading cycle')
    # plt.plot(last_cycle_plot['last_cycle_disp'], last_cycle_plot['last_cycle_force'], label='regression area', color='k')
    plt.plot(last_cycle_plot['last_cycle_disp'], stiffness * last_cycle_plot['last_cycle_disp'] +
             intercept_value_max_stiff, label='Fit', color='black')
    plt.plot([], ' ', label=f'Stiffness = {stiffness:.0f} N/mm')
    plt.plot([], ' ', label='Sample ID: ' + sample_ID)
    plt.ylabel('force lc / N')
    plt.xlabel('disp ext / mm')
    plt.legend()
    plt.autoscale()
    plt.rcParams.update({'font.size': 14})
    # plt.legend(prop={'size': 14})
    plt.savefig(os.path.join(savepath_fd, 'force_disp_el_' + sample_ID + '.png'), dpi=300, bbox_inches='tight', format='png')
    # plt.show()
    plt.close()

    # calculate stress/strain, filter and put into dataframe
    l_initial = 6.5
    mean_area_wop = results_uCT['Mean Apparent Area / mm^2'][counter]
    mean_bone_area_wp = results_uCT['Mean ECM Area / mm^2'][counter]
    stress_wop = df['force_lc'] / mean_area_wop
    stress_bone_wp = df['force_lc'] / mean_bone_area_wp
    strain = df['disp_ext'] / l_initial
    df['stress_lc_wop'] = stress_wop
    df['stress_bone_wp'] = stress_bone_wp
    df['strain_ext'] = strain
    df['stress_lc_filtered_wop'] = butter_lowpass_filter(df['stress_lc_wop'], cutoff)
    df['stress_bone_wp_filtered'] = butter_lowpass_filter(df['stress_bone_wp'], cutoff)
    counter = counter + 1

    # plot stress/strain
    plt.figure()
    plt.title(sample_ID)
    plt.plot(df['strain_ext'], df['stress_lc_wop'], label='raw')
    plt.plot(df['strain_ext'], df['stress_lc_filtered_wop'], label='filtered')
    plt.ylabel('stress / MPa')
    plt.xlabel('strain / -')
    plt.legend()
    # plt.show()
    # savepath = Cwd / '04_Results/00_Mineralized/01_stress_strain/'
    # plt.savefig(os.path.join(savepath, 'stress_strain_' + sample_ID + '.png'), dpi=300)
    plt.close()

    # calculate values for last cycle of stress/strain curve only
    last_cycle_stress = df['stress_lc_filtered_wop'][peaks_index[-1]:]
    last_cycle_strain = df['strain_ext'][peaks_index[-1]:]
    last_cycle_strain = last_cycle_strain.dropna().reset_index(drop=True)
    last_cycle_stress = last_cycle_stress.dropna().reset_index(drop=True)

    last_cycle_stress_bone = df['stress_bone_wp_filtered'][peaks_index[-1]:]
    last_cycle_stress_bone = last_cycle_stress_bone.dropna().reset_index(drop=True)

    # plot last cycle of stress/strain curve
    plt.figure()
    plt.title(sample_ID)
    plt.plot(last_cycle_strain, last_cycle_stress)
    plt.ylabel('stress / MPa')
    plt.xlabel('strain / -')
    # plt.show()
    plt.close()

    ## calculate apparent modulus by using rolling regression
    # isolate stress/strain values of defined strain region
    last_cycle = pd.DataFrame()
    last_cycle_sb = pd.DataFrame()
    last_cycle['last_cycle_strain'] = round(last_cycle_strain, 5)
    last_cycle['last_cycle_stress'] = round(last_cycle_stress, 5)
    last_cycle_sb['last_cycle_strain_sb'] = round(last_cycle_strain, 5)
    last_cycle_sb['last_cycle_stress_sb'] = round(last_cycle_stress_bone, 3)

    # identify index range of defined region, exact upper_strain value not found
    if 0.00250 in last_cycle['last_cycle_strain'].values:
        upper_cond = last_cycle.loc[last_cycle['last_cycle_strain'] == 0.00250]
    elif 0.00249 in last_cycle['last_cycle_strain'].values:
        upper_cond = last_cycle.loc[last_cycle['last_cycle_strain'] == 0.00249]
    elif 0.00251 in last_cycle['last_cycle_strain'].values:
        upper_cond = last_cycle.loc[last_cycle['last_cycle_strain'] == 0.00251]

    if 0.00100 in last_cycle['last_cycle_strain'].values:
        lower_cond = last_cycle.loc[last_cycle['last_cycle_strain'] == 0.00100]
    elif 0.0009 in last_cycle['last_cycle_strain'].values:
        lower_cond = last_cycle.loc[last_cycle['last_cycle_strain'] == 0.00090]
    elif 0.00110 in last_cycle['last_cycle_strain'].values:
        lower_cond = last_cycle.loc[last_cycle['last_cycle_strain'] == 0.00110]

    max_strain_ind = min(upper_cond.index)
    min_strain_ind = min(lower_cond.index)
    last_cycle = last_cycle[max_strain_ind:min_strain_ind]
    last_cycle_sb = last_cycle_sb[max_strain_ind:min_strain_ind]

    # initialize lists for slope/intercept value collection
    slope_values_app = list()
    intercept_values_app = list()
    slope_values_sb = list()
    intercept_values_sb = list()

    # rolling linear regression for apparent modulus calculation
    for x in range(max_strain_ind, min_strain_ind - window_width + 1, 1):
        last_cycle_mod = last_cycle[x:x+window_width]
        last_cycle_mod_sb = last_cycle_sb[x:x+window_width]
        slope_app, intercept_app, r_value, p_value, std_err = stats.linregress(last_cycle_mod['last_cycle_strain'],
                                                                               last_cycle_mod['last_cycle_stress'])
        slope_sb, intercept_sb, r_value_sb, p_value_sb, std_err_sb = stats.linregress(last_cycle_mod_sb['last_cycle_strain_sb'],
                                                                                      last_cycle_mod_sb['last_cycle_stress_sb'])
        # collect slope value & add to growing list; same for intercept
        slope_value_app = slope_app
        slope_values_app.append(slope_value_app)
        intercept_value_app = intercept_app
        intercept_values_app.append(intercept_value_app)

        slope_value_sb = slope_sb
        slope_values_sb.append(slope_value_sb)
        intercept_value_sb = intercept_sb
        intercept_values_sb.append(intercept_value_sb)


    # Create DataFrames for slope or stiffness & intercept to search for max. values & indices; create cycle for plotting
    slope_value_app_df = pd.DataFrame(slope_values_app)
    intercept_values_app_df = pd.DataFrame(intercept_values_app)
    apparent_modulus = slope_value_app_df.max()[0]
    max_slope_index_app = slope_value_app_df[slope_value_app_df == apparent_modulus].dropna()
    max_slope_index_app = max_slope_index_app.index
    intercept_value_max_app = intercept_values_app_df.loc[max_slope_index_app[0]].values[0]
    last_cycle_plot = last_cycle[max_slope_index_app[0]:max_slope_index_app[0] + window_width]

    slope_value_sb_df = pd.DataFrame(slope_values_sb)
    intercept_values_sb_df = pd.DataFrame(intercept_values_sb)
    modulus_sb = slope_value_sb_df.max()[0]
    max_slope_index_sb = slope_value_sb_df[slope_value_sb_df == modulus_sb].dropna()
    max_slope_index_sb = max_slope_index_sb.index
    intercept_value_max_sb = intercept_values_sb_df.loc[max_slope_index_sb[0]].values[0]
    last_cycle_plot = last_cycle[max_slope_index_sb[0]:max_slope_index_sb[0] + window_width]

    # generate plot
    # plt.figure(figsize=(6, 4))
    # plt.title(sample_ID)
    plt.plot(df['strain_ext'], df['stress_lc_filtered_wop'], label='Filtered')
    plt.plot(last_cycle['last_cycle_strain'], last_cycle['last_cycle_stress'], label='Last unloading cycle')
    # plt.plot(last_cycle_plot['last_cycle_strain'], last_cycle_plot['last_cycle_stress'], label='regress area', color='k')
    plt.plot(last_cycle_plot['last_cycle_strain'], apparent_modulus * last_cycle_plot['last_cycle_strain'] +
             intercept_value_max_app, label='Fit', color='black')
    plt.plot([], ' ', label=f'Apparent modulus = {apparent_modulus:.0f} MPa')
    plt.plot([], ' ', label='Sample ID: ' + sample_ID)
    plt.ylabel('stress / MPa')
    plt.xlabel('strain / -')
    plt.legend()
    plt.autoscale()
    plt.rcParams.update({'font.size': 14})
    # plt.legend(prop={'size': 14})
    plt.savefig(os.path.join(savepath_ss, 'stress_strain_el_' + sample_ID + '.png'), dpi=300, bbox_inches='tight', format='png')
    # plt.show()
    plt.close()

    # create list with current values which are sample_ID, slope & apparent modulus & add them to result list which
    # is then converted to dataframe
    values = [sample_ID, round(stiffness), round(apparent_modulus), round(modulus_sb)]
    result.append(values)
    result_dir = pd.DataFrame(result, columns=['Sample ID', 'Stiffness N/mm', 'Apparent modulus MPa', 'Modulus Mineralized MPa'])

    # Create displacement-force-time plots
    rcParams.update({'figure.autolayout': True})
    time = pd.DataFrame()
    time = df['time'] - df['time'].loc[0]
    fig, ax1 = plt.subplots()
    ax1.plot(time, df['disp_ext'], label='Displacement')
    ax2 = ax1.twinx()
    ax2.plot(time, df['force_lc'], color='darkorange', label='Force')
    plt.plot([], ' ', label='Sample ID: ' + sample_ID)
    # plt.title(sample_ID)
    ax1.set_xlabel('Time s')
    ax1.set_ylabel('Displacement mm')
    ax2.set_ylabel('Force N')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=3)
    ax1.autoscale()
    ax2.autoscale()
    plt.rcParams.update({'font.size': 14})
    plt.savefig(os.path.join(savepath_dft, 'disp_time_el_' + sample_ID + '.png'), dpi=300, bbox_inches='tight',
                format='png')
    # plt.show()
    plt.close()

# add missing samples to list & save
missing_sample_IDs = pd.DataFrame({'Sample ID': ['390R', '395R', '400R', '402L', '410L', '433L']})
result_dir = pd.concat([result_dir, missing_sample_IDs])
result_dir_sorted = result_dir.sort_values(by=['Sample ID'], ascending=True)

result_dir_sorted.to_csv(os.path.join(str(os.path.dirname(Cwd)) + '/04_Results/00_Mineralized/',
                                      'ResultsElasticTesting.csv'), index=False)
