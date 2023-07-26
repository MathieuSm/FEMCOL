# This script plots various variables against each other, Data is retrieved from ResultsOverview.csv file

# Import standard packages
import pandas as pd                                 # Used to manage data frames
import matplotlib.pyplot as plt                     # Used to perform plots
import os                                           # Used to manage path variables


# Set directory & load data
Cwd = os.getcwd()
results_path = str(os.path.dirname(Cwd) + '/04_Results')
results_overview = str(os.path.dirname(Cwd) + '/04_Results/ResultsOverview.csv')
SavePath = results_path + '/04_Plots'
df = pd.read_csv(str(results_overview), skiprows=0)

E_m = (df['Modulus Mineralized / MPa'].dropna().reset_index(drop=True))/1000
E_app_m = (df['Apparent Modulus Mineralized / MPa'].dropna().reset_index(drop=True))/1000

E_c = df['Modulus Demineralized / MPa'].dropna().reset_index(drop=True)
E_app_c = df['Apparent Modulus Demineralized / MPa'].dropna().reset_index(drop=True)

k_m = (df['Stiffness Mineralized / N/mm'].dropna().reset_index(drop=True))/1000
k_c = (df['Stiffness Demineralized / N/mm'].dropna().reset_index(drop=True))/1000

sigma_u = df['Ultimate Stress / MPa'].dropna().reset_index(drop=True)
sigma_app_u = df['Ultimate Apparent Stress / MPa'].dropna().reset_index(drop=True)
sigma_u_col = df['Ultimate Collagen Stress / MPa'].dropna().reset_index(drop=True)

f_u = df['Ultimate Force / N'].dropna().reset_index(drop=True)

epsilon_u = df['Ultimate Strain / -'].dropna().reset_index(drop=True)

d_b = df['Density / g/cm³'].dropna().reset_index(drop=True)

Wo = df['Organic Weight / g'].dropna().reset_index(drop=True)
Wm = df['Mineral Weight / g'].dropna().reset_index(drop=True)
Ww = df['Water Weight / g'].dropna().reset_index(drop=True)

WFo = df['Organic weight fraction / -'].dropna().reset_index(drop=True)
WFm = df['Mineral weight fraction / -'].dropna().reset_index(drop=True)
WFw = df['Water weight fraction / -'].dropna().reset_index(drop=True)

BVTV = df['Bone Volume Fraction / -'].dropna().reset_index(drop=True)

BMD = df['Bone Mineral Density / mg HA / cm³'].dropna().reset_index(drop=True)
TMD = df['Tissue Mineral Density / mg HA / cm³'].dropna().reset_index(drop=True)

BMC = df['Bone Mineral Content / mg HA'].dropna().reset_index(drop=True)

MMRV2A3 = df['Mineral to Matrix Ratio v2/a3 / -'].dropna().reset_index(drop=True)
MMRV1A1 = df['Mineral to Matrix Ratio v1/a1 / -'].dropna().reset_index(drop=True)

Xc = df['Crystallinity / -'].dropna().reset_index(drop=True)
RPyC = df['Relative Pyridinoline Content / -'].dropna().reset_index(drop=True)
RPrC = df['Relative Proteoglycan Content / -'].dropna().reset_index(drop=True)
RLC = df['Relative Lipid Content / -'].dropna().reset_index(drop=True)

COLDIS = df['Collagen dis/order / -'].dropna().reset_index(drop=True)

MATMAT = df['Matrix maturity / -'].dropna().reset_index(drop=True)


columns1 = [E_m, E_app_m]
columns2 = [E_c, E_app_c]
columns3 = [k_m, k_c]
columns4 = [sigma_u, sigma_app_u, sigma_u_col]
columns5 = [f_u]
columns6 = [epsilon_u]
columns7 = [d_b]
columns8 = [Wo, Wm, Ww]
columns9 = [BVTV]
columns10 = [BMD, TMD]
columns11 = [BMC]
columns12 = [MMRV2A3, MMRV1A1]
columns13 = [Xc, RPyC, RPrC, RLC]
columns14 = [COLDIS]
columns15 = [MATMAT]
columns16 = [WFo, WFm, WFw]


YposCI = 0.025
YposCV = YposCI + 0.075
YposN = YposCV + 0.075

plt.rcParams["font.size"] = "6"
plt.rcParams['lines.linewidth'] = 1

flierprops = {'markersize': 1}
boxprops={'linewidth': 0.5}
whiskerprops={'linewidth': 0.5}
capprops={'linewidth': 0.5}
meanprops={'linewidth': 0.5}
medianprops={'linewidth': 0.5}

# fist subplot
plt.figure()
plt.subplot(4, 4, 1)
box = plt.boxplot(columns1, patch_artist=True, showmeans=True, meanline=True, flierprops=flierprops,
                  boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanprops,
                  medianprops=medianprops)
plt.tight_layout()
plt.xticks([1, 2], ['E$_{app, m}$', 'E$_m$'])
colors = ['lightblue', 'lightgreen']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('GPa')

# min = columns1[0].min()
# max = columns1[0].max()
# mean = stats.mean(columns1[0])
# median = stats.median(columns1[0])
# std = stats.stdev(columns1[0])
# cv = std/mean

# plt.annotate(r'$\downarrow$: ' + str(min) + r', $\uparrow$:'  + str(max), xy=(0.05, YposN), xycoords='axes fraction', fontsize=5)
# plt.annotate(r'$mean \pm std$ : ' + str(mean) + ', $\pm $' + str(round(std,1)), xy=(0.05, YposCV), xycoords='axes fraction', fontsize=5)
# plt.annotate(r'$CV$ : ' + str(round(cv, 3)), xy=(0.05, YposCI), xycoords='axes fraction', fontsize=5)

# second subplot
plt.subplot(4, 4, 2)
box = plt.boxplot(columns2, patch_artist=True, showmeans=True, meanline=True, flierprops={'markersize': 1},
                  boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanprops,
                  medianprops=medianprops)
plt.tight_layout()
plt.xticks([1, 2], ['E$_{app, c}$', 'E$_c$'])
colors = ['lightblue', 'lightgreen']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('MPa')

plt.subplot(4, 4, 3)
box = plt.boxplot(columns3, patch_artist=True, showmeans=True, meanline=True, flierprops={'markersize': 1},
                  boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanprops,
                  medianprops=medianprops)
plt.tight_layout()
plt.xticks([1, 2], ['k$_{m}$', 'k$_{c}$'])
colors = ['lightblue', 'lightgreen']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('kN/mm')

plt.subplot(4, 4, 4)
box = plt.boxplot(columns4, patch_artist=True, showmeans=True, meanline=True, flierprops={'markersize': 1},
                  boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanprops,
                  medianprops=medianprops)
plt.tight_layout()
plt.xticks([1, 2, 3], ['$\sigma_u$', '$\sigma_{u_{app}}$', '$\sigma_{u_{col}}$'])
colors = ['lightblue', 'lightgreen', 'lightpink']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('MPa')

plt.subplot(4, 4, 5)
box = plt.boxplot(columns5, patch_artist=True, showmeans=True, meanline=True, flierprops={'markersize': 1},
                  boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanprops,
                  medianprops=medianprops)
plt.tight_layout()
plt.xticks([1], ['F$_{u}$'])
colors = ['lightblue']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('N')

plt.subplot(4, 4, 6)
box = plt.boxplot(columns6, patch_artist=True, showmeans=True, meanline=True, flierprops={'markersize': 1},
                  boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanprops,
                  medianprops=medianprops)
plt.tight_layout()
plt.xticks([1], ['$\epsilon_{u}$'])
colors = ['lightblue']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('-')

plt.subplot(4, 4, 7)
box = plt.boxplot(columns7, patch_artist=True, showmeans=True, meanline=True, flierprops={'markersize': 1},
                  boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanprops,
                  medianprops=medianprops)
plt.tight_layout()
plt.xticks([1], [r'$\rho_{b}$'])
colors = ['lightblue']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('g / cm³')

plt.subplot(4, 4, 8)
box = plt.boxplot(columns8, patch_artist=True, showmeans=True, meanline=True, flierprops={'markersize': 1},
                  boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanprops,
                  medianprops=medianprops)
plt.tight_layout()
plt.xticks([1, 2, 3], ['m$_o$', 'm$_m$', 'm$_w$'])
colors = ['lightblue', 'lightgreen', 'lightpink']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('g')

plt.subplot(4, 4, 9)
box = plt.boxplot(columns9, patch_artist=True, showmeans=True, meanline=True, flierprops={'markersize': 1},
                  boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanprops,
                  medianprops=medianprops)
plt.tight_layout()
plt.xticks([1], ['BVTV'])
colors = ['lightblue']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('-')

plt.subplot(4, 4, 10)
box = plt.boxplot(columns10, patch_artist=True, showmeans=True, meanline=True, flierprops={'markersize': 1},
                  boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanprops,
                  medianprops=medianprops)
plt.tight_layout()
plt.xticks([1, 2], ['BMD', 'TMD'])
colors = ['lightblue', 'lightgreen']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('mg HA / cm³')

plt.subplot(4, 4, 11)
box = plt.boxplot(columns11, patch_artist=True, showmeans=True, meanline=True, flierprops={'markersize': 1},
                  boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanprops,
                  medianprops=medianprops)
plt.tight_layout()
plt.xticks([1], ['BMC'])
colors = ['lightblue']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('mg HA')

plt.subplot(4, 4, 12)
box = plt.boxplot(columns12, patch_artist=True, showmeans=True, meanline=True, flierprops={'markersize': 1},
                  boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanprops,
                  medianprops=medianprops)
plt.tight_layout()
plt.xticks([1, 2], [r'$MMR_{\nu_{2}a_{3}}$', r'$MMR_{\nu_{1}a_{1}}$'])
colors = ['lightblue', 'lightgreen']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('-')

plt.subplot(4, 4, 13)
box = plt.boxplot(columns13, patch_artist=True, showmeans=True, meanline=True, flierprops={'markersize': 1},
                  boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanprops,
                  medianprops=medianprops)
plt.tight_layout()
plt.xticks([1, 2, 3, 4], ['X$_{c}$', 'RPyC', 'RPrC', 'RLC'])
colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('-')

plt.subplot(4, 4, 14)
box = plt.boxplot(columns14, patch_artist=True, showmeans=True, meanline=True, flierprops={'markersize': 1},
                  boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanprops,
                  medianprops=medianprops)
plt.tight_layout()
plt.xticks([1], ['ColDis'])
colors = ['lightblue']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('-')

plt.subplot(4, 4, 15)
box = plt.boxplot(columns15, patch_artist=True, showmeans=True, meanline=True, flierprops={'markersize': 1},
                  boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanprops,
                  medianprops=medianprops)
plt.tight_layout()
plt.xticks([1], ['MatMat'])
colors = ['lightblue']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('-')

plt.subplot(4, 4, 16)
box = plt.boxplot(columns16, patch_artist=True, showmeans=True, meanline=True, flierprops={'markersize': 1},
                  boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanprops,
                  medianprops=medianprops)
plt.tight_layout()
plt.xticks([1, 2, 3], ['WF$_o$', 'WF$_m$', 'WF$_w$'])
colors = ['lightblue', 'lightgreen', 'lightpink']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# ax.set_title('Example Boxplot')
plt.ylabel('-')

plt.savefig(os.path.join(SavePath, 'Overview_Boxplots.png'), dpi=1200, format='png')
plt.show()
