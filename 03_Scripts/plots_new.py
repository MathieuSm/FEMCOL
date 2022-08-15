# This script plots various variables against each other, Data is retrieved from ResultsOverview.csv file

# Import standard packages
from pathlib import Path                 # Used to manage path variables in windows or linux
import numpy as np                       # Used to do arrays (matrices) computations namely
import pandas as pd                      # Used to manage data frames
import matplotlib.pyplot as plt          # Used to perform plots
import statsmodels.formula.api as smf    # Used for statistical analysis (ols here)
import os
from scipy.stats.distributions import t  # Used to compute confidence intervals
import sys
import seaborn as sns

# Set directory & load data
Cwd = Path.cwd()
DataPath = Cwd / '04_Results/ResultsOverview.csv'
# DataPath = 'C:/Users/Stefan/Dropbox/02_MScThesis/09_Results/ResultsOverview.csv'
df = pd.read_csv(str(DataPath), skiprows=0)
SampleID = df['Sample ID'].values.tolist()

# Create dataframe with variable names & respective abbreviations
ColumnNames = pd.DataFrame()
ColumnNames['Column Names'] = df.columns
column_names_abbrev = ['SID', 'Age', 'G', 'SM', 'AMM', 'UF', 'USTRE', 'USTRA', 'SD', 'AMD', 'D', 'OW', 'MW', 'WW',
                       'MWF', 'OWF', 'WWF', 'BVTV', 'BMD', 'TMD', 'BMC', 'MINA', 'MEANAA', 'MINED', 'MEANAD', 'MEANAF',
                       'MINAF', 'MMR']
ColumnNames['Abbreviations'] = column_names_abbrev

Pair = pd.DataFrame([
                     ['Age',                                'Apparent Modulus Demineralized MPa'],
                     ['Age',                                'Apparent Modulus Mineralized MPa'],
                     ['Age',                                'Bone Mineral Content mg HA'],
                     ['Age',                                'Bone Mineral Density mg HA / cm³'],
                     ['Age',                                'Bone Volume Fraction -'],
                     ['Age',                                'Density g/cm³'],
                     ['Age',                                'Mineral Weight g'],
                     ['Age',                                'Mineral weight fraction -'],
                     ['Age',                                'Organic Weight g'],
                     ['Age',                                'Organic weight fraction -'],
                     ['Age',                                'Stiffness Demineralized N/mm'],
                     ['Age',                                'Stiffness Mineralized N/mm'],
                     ['Age',                                'Tissue Mineral Density mg HA / cm³'],
                     ['Age',                                'Ultimate Force N'],
                     ['Age',                                'Ultimate Strain -'],
                     ['Age',                                'Ultimate Stress MPa'],
                     ['Age',                                'Water Weight g'],
                     ['Age',                                'Water weight fraction -'],
                     ['Age',                                'Mean Apparent Area mm²'],
                     ['Age',                                'Minimum Area mm²'],
                     ['Age',                                'Mean Area Fraction -'],
                     ['Age',                                'Min Area Fraction -'],
                     ['Bone Mineral Content mg HA',         'Stiffness Mineralized N/mm'],
                     ['Bone Mineral Density mg HA / cm³',   'Apparent Modulus Mineralized MPa'],
                     ['Bone Mineral Density mg HA / cm³',   'Ultimate Stress MPa'],
                     ['Bone Volume Fraction -',             'Apparent Modulus Mineralized MPa'],
                     ['Bone Volume Fraction -',             'Ultimate Stress MPa'],
                     ['Apparent Modulus Mineralized MPa',   'Apparent Modulus Demineralized MPa'],
                     ['Mean Apparent Area mm²',             'Apparent Modulus Demineralized MPa'],
                     # ['Mean Apparent Diameter mm',        'Apparent Modulus Mineralized MPa'],
                     ['Organic weight fraction -',          'Ultimate Stress MPa'],
                     # ['Organic Weight g',                 'Ultimate Stress MPa'],
                     ['Organic weight fraction -',          'Stiffness Mineralized N/mm'],
                     ['Organic weight fraction -',          'Stiffness Demineralized N/mm'],
                     ['Min Area Fraction -',                'Ultimate Stress MPa'],
                     ['Mineral weight fraction -',          'Stiffness Demineralized N/mm'],
                     ['Mineral weight fraction -',          'Stiffness Mineralized N/mm'],
                     ['Mineral weight fraction -',          'Ultimate Stress MPa'],
                     ['Bone Mineral Content mg HA',         'Mineral Weight g'],
                     ['Mean Apparent Area mm²',             'Ultimate Stress MPa'],
                     ['Apparent Modulus Demineralized MPa', 'Ultimate Stress MPa'],
                     ['Mineral weight fraction -',          'Mineral to Matrix Ratio -']
#                      ['Bone Volume Fraction -',             'Bone Mineral Density mg HA / cm³'],
#                      ['Bone Volume Fraction -',             'Tissue Mineral Density mg HA / cm³'],
#                      ['Bone Volume Fraction -',             'Mineral weight fraction -'],
#                      ['Bone Volume Fraction -',             'Organic weight fraction -'],
#                      ['Bone Volume Fraction -',             'Water weight fraction -'],
#                      ['Bone Volume Fraction -',             'Density g/cm³'],
#                      ['Bone Volume Fraction -',             'Apparent Modulus Mineralized MPa'],
#                      ['Bone Volume Fraction -',             'Apparent Modulus Demineralized MPa'],
#                      ['Bone Volume Fraction -',             'Ultimate Stress MPa'],
#                      ['Bone Volume Fraction -',             'Ultimate Strain -'],
#                      ['Bone Volume Fraction -',             'Mineral to Matrix Ratio -'],
#                      ['Bone Mineral Density mg HA / cm³',             'Tissue Mineral Density mg HA / cm³'],
#                      ['Bone Mineral Density mg HA / cm³',             'Mineral weight fraction -'],
#                      ['Bone Mineral Density mg HA / cm³',             'Organic weight fraction -'],
#                      ['Bone Mineral Density mg HA / cm³',             'Water weight fraction -'],
#                      ['Bone Mineral Density mg HA / cm³',             'Density g/cm³'],
#                      ['Bone Mineral Density mg HA / cm³',             'Apparent Modulus Mineralized MPa'],
#                      ['Bone Mineral Density mg HA / cm³',             'Apparent Modulus Demineralized MPa'],
#                      ['Bone Mineral Density mg HA / cm³',             'Ultimate Stress MPa'],
#                      ['Bone Mineral Density mg HA / cm³',             'Ultimate Strain -'],
#                      ['Bone Mineral Density mg HA / cm³',             'Mineral to Matrix Ratio -'],
#                      ['Tissue Mineral Density mg HA / cm³',             'Mineral weight fraction -'],
#                      ['Tissue Mineral Density mg HA / cm³',             'Organic weight fraction -'],
#                      ['Tissue Mineral Density mg HA / cm³',             'Water weight fraction -'],
#                      ['Tissue Mineral Density mg HA / cm³',             'Density g/cm³'],
#                      ['Tissue Mineral Density mg HA / cm³',             'Apparent Modulus Mineralized MPa'],
#                      ['Tissue Mineral Density mg HA / cm³',             'Apparent Modulus Demineralized MPa'],
#                      ['Tissue Mineral Density mg HA / cm³',             'Ultimate Stress MPa'],
#                      ['Tissue Mineral Density mg HA / cm³',             'Ultimate Strain -'],
#                      ['Tissue Mineral Density mg HA / cm³',             'Mineral to Matrix Ratio -'],
#                      ['Mineral weight fraction -',             'Organic weight fraction -'],
#                      ['Mineral weight fraction -',             'Water weight fraction -'],
#                      ['Mineral weight fraction -',             'Density g/cm³'],
#                      ['Mineral weight fraction -',             'Apparent Modulus Mineralized MPa'],
#                      ['Mineral weight fraction -',             'Apparent Modulus Demineralized MPa'],
#                      ['Mineral weight fraction -',             'Ultimate Stress MPa'],
#                      ['Mineral weight fraction -',             'Ultimate Strain -'],
#                      ['Mineral weight fraction -',             'Mineral to Matrix Ratio -'],
#                      ['Organic weight fraction -',             'Water weight fraction -'],
#                      ['Organic weight fraction -',             'Density g/cm³'],
#                      ['Organic weight fraction -',             'Apparent Modulus Mineralized MPa'],
#                      ['Organic weight fraction -',             'Apparent Modulus Demineralized MPa'],
#                      ['Organic weight fraction -',             'Ultimate Stress MPa'],
#                      ['Organic weight fraction -',             'Ultimate Strain -'],
#                      ['Organic weight fraction -',             'Mineral to Matrix Ratio -'],
#                      ['Water weight fraction -',             'Density g/cm³'],
#                      ['Water weight fraction -',             'Apparent Modulus Mineralized MPa'],
#                      ['Water weight fraction -',             'Apparent Modulus Demineralized MPa'],
#                      ['Water weight fraction -',             'Ultimate Stress MPa'],
#                      ['Water weight fraction -',             'Ultimate Strain -'],
#                      ['Water weight fraction -',             'Mineral to Matrix Ratio -'],
#                      ['Density g/cm³',             'Apparent Modulus Mineralized MPa'],
#                      ['Density g/cm³',             'Apparent Modulus Demineralized MPa'],
#                      ['Density g/cm³',             'Ultimate Stress MPa'],
#                      ['Density g/cm³',             'Ultimate Strain -'],
#                      ['Density g/cm³',             'Mineral to Matrix Ratio -'],
#                      ['Apparent Modulus Mineralized MPa',             'Apparent Modulus Demineralized MPa'],
#                      ['Apparent Modulus Mineralized MPa',             'Ultimate Stress MPa'],
#                      ['Apparent Modulus Mineralized MPa',             'Ultimate Strain -'],
#                      ['Apparent Modulus Demineralized MPa',             'Ultimate Stress MPa'],
#                      ['Apparent Modulus Demineralized MPa',             'Ultimate Strain -'],
#                      ['Apparent Modulus Demineralized MPa',             'Mineral to Matrix Ratio -'],
#                      ['Ultimate Stress MPa',                'Ultimate Strain -'],
#                      ['Ultimate Stress MPa',                'Mineral to Matrix Ratio -'],
                     ])

# assign abbreviations to above list of variables
Pair_abbrev1 = list()
Pair_abbrev2 = list()

for i in range(len(Pair)):
    index1 = ColumnNames.loc[ColumnNames['Column Names'] == Pair[0][i]].index[0]
    index2 = ColumnNames.loc[ColumnNames['Column Names'] == Pair[1][i]].index[0]
    Abbrev1 = ColumnNames['Abbreviations'][index1]
    Abbrev2 = ColumnNames['Abbreviations'][index2]
    Pair_abbrev1.append(Abbrev1)
    Pair_abbrev2.append(Abbrev2)

Pair_abbrev1_df = pd.DataFrame(Pair_abbrev1)
Pair_abbrev2_df = pd.DataFrame(Pair_abbrev2)
Pair_abbrev_df = pd.DataFrame()
Pair_abbrev_df['Abbrev_x'] = Pair_abbrev1_df
Pair_abbrev_df['Abbrev_y'] = Pair_abbrev2_df

# loop to iterate through lists of names & create plots
results = list()
j = 0

for i in range(len(Pair)):
    x_axis = Pair[0][i]
    y_axis = Pair[1][i]
    x_axis_abbrev = Pair_abbrev_df['Abbrev_x'][i]
    y_axis_abbrev = Pair_abbrev_df['Abbrev_y'][i]
    Data = df.filter(['Sample ID', x_axis, y_axis, 'Gender', 'Age']).dropna()
    Data2Fit = Data.copy()
    Data2Fit.rename(columns={'Sample ID': 'SID', x_axis: x_axis_abbrev, y_axis: y_axis_abbrev}, inplace=True)
    Data2Fit = Data2Fit.set_index('SID')
    FitResults = smf.ols(y_axis_abbrev + ' ~ 1 + ' + x_axis_abbrev, data=Data2Fit).fit()

    # Calculate R^2, p-value, 95% CI, SE, N
    Y_Obs = FitResults.model.endog
    Y_Fit = FitResults.fittedvalues

    E = Y_Obs - Y_Fit
    RSS = np.sum(E ** 2)
    SE = np.sqrt(RSS / FitResults.df_resid)
    CI_l = FitResults.conf_int()[0][1]
    CI_r = FitResults.conf_int()[1][1]
    N = int(FitResults.nobs)
    R2 = FitResults.rsquared
    p = FitResults.pvalues[1]
    X = np.matrix(FitResults.model.exog)
    X_Obs = np.sort(np.array(X[:, 1]).reshape(len(X)))
    C = np.matrix(FitResults.cov_params())
    B_0 = np.sqrt(np.diag(np.abs(X * C * X.T)))
    Alpha = 0.95
    t_Alpha = t.interval(Alpha, N - X.shape[1] - 1)
    CI_Line_u = Y_Fit + t_Alpha[0] * SE * B_0
    CI_Line_o = Y_Fit + t_Alpha[1] * SE * B_0
    Sorted_CI_u = CI_Line_u[np.argsort(FitResults.model.exog[:,1])]
    Sorted_CI_o = CI_Line_o[np.argsort(FitResults.model.exog[:,1])]

    savepath = Cwd / '04_Results/04_Plots/'
    savepathCorrMat = Cwd / '04_Results/04_Plots/CorrelationMatrix'
    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=300, sharey=True, sharex=True)
    male_age = Data[Data['Gender'] == 'M']['Age']
    female_age = Data[Data['Gender'] == 'F']['Age']
    X_np = np.array(X)
    Y_Obs_np = np.array(Y_Obs)

    # Requirements for rounding: depending on value
    if abs(CI_l) >= 100:
        CI_l = round(CI_l)
    elif abs(CI_l) >= 1:
        CI_l = round(CI_l, 1)
    elif abs(CI_l) == 0:
        CI_l = int(CI_l)
    elif abs(CI_l) >= 0.001:
        CI_l = round(CI_l, 3)
    elif abs(CI_l) < 0.001:
        CI_l = '{:.2e}'.format(CI_l)

    if abs(CI_r) >= 100:
        CI_r = round(CI_r)
    elif abs(CI_r) >= 1:
        CI_r = round(CI_r, 1)
    elif abs(CI_r) == 0:
        CI_r = int(CI_r)
    elif abs(CI_r) >= 0.001:
        CI_r = round(CI_r, 3)
    elif abs(CI_r) < 0.001:
        CI_r = '{:.2e}'.format(CI_r)

    if abs(SE) >= 100:
        SE = round(SE)
    elif abs(SE) >= 1:
        SE = round(SE, 1)
    elif abs(SE) == 0:
        SE = int(SE)
    elif abs(SE) >= 0.001:
        SE = round(SE, 3)
    elif abs(SE) < 0.001:
        SE = '{:.1e}'.format(SE)

    if p < 0.001:
        p = '{:.1e}'.format(p)
    else:
        p = round(p, 3)

    if float(R2) < 0.01:
        R2 = float(R2)
        R2 = '{:.1e}'.format(R2)
    else:
        R2 = float(R2)
        R2 = round(R2, 3)

    # list of plots which need autoscaling, has to be ordered manually
    autoscale_list = pd.DataFrame({'x_axis_abbrev': ['Age',  'Age', 'Age', 'Age', 'Age', 'Age',    'Age',  'Age',
                                                        'Age',   'Age', 'BMC'],
                                   'y_axis_abbrev': ['BMD', 'BVTV',   'D', 'MWF', 'OWF', 'TMD', 'MEANAA', 'MINA',
                                                     'MEANAF', 'MINAF', 'MWF']})
    # if p-value smaller than 0.05 create fit curve and if variable 'Age' should not be plotted on main axis, no
    # colormap will be used
    if float(p) <= 0.05:
        if x_axis != 'Age':
            # sns.regplot(x=FitResults.model.exog[:,1], y=Y_Obs, ax=Axes, scatter=False)
            Axes.plot(X[:, 1], Y_Fit, color=(1, 0, 0), label='Fit')
            # Axes.fill_between(X_Obs, Sorted_CI_o, Sorted_CI_u, color=(0, 0, 0), alpha=0.1,
            #                   label=str(int(Alpha * 100)) + '% CI')
            Axes.scatter(X_np[:, 1][Data['Gender'] == 'M'], Y_Obs_np[Data['Gender'] == 'M'],
                         c=list(tuple(male_age.tolist())), cmap='plasma_r', vmin=Data['Age'].min(),
                         vmax=Data['Age'].max(), label='male', marker='s')
            Axes.scatter(X_np[:, 1][Data['Gender'] == 'F'], Y_Obs_np[Data['Gender'] == 'F'],
                         c=list(tuple(female_age.tolist())), cmap='plasma_r', vmin=Data['Age'].min(),
                         vmax=Data['Age'].max(), label='female', marker='o')
            Regression_line = FitResults.params[1] * X_np[:, 1] + FitResults.params[0]
            ax = plt.gca()
            PCM = ax.get_children()[2]
            plt.colorbar(PCM, ax=ax, label='Age y')
            Axes.annotate(r'$N$ : ' + str(N), xy=(0.05, 0.325), xycoords='axes fraction')
            Axes.annotate(r'$R^2$ : ' + str(R2), xy=(0.05, 0.25), xycoords='axes fraction')
            Axes.annotate(r'$\sigma_{est}$ : ' + str(SE), xy=(0.05, 0.175), xycoords='axes fraction')
            Axes.annotate(r'$p$ : ' + str(p), xy=(0.05, 0.1), xycoords='axes fraction')
            Axes.annotate('95% CI [' + str(CI_l) + r'$,$ ' + str(CI_r) + ']', xy=(0.05, 0.025),
                          xycoords='axes fraction')
            Axes.set_ylabel(Data.columns[2])
            Axes.set_xlabel(Data.columns[1])

            # condition used for autoscaling
            if x_axis_abbrev == autoscale_list.loc[j][0] and y_axis_abbrev == autoscale_list.loc[j][1]:
                # plt.ylim(ymin=0, ymax=round(Y_Fit.max() * 1.2, 2))
                plt.autoscale()
                plt.subplots_adjust(left=0.15, bottom=0.15)
                plt.legend(loc='upper center', bbox_to_anchor=(0,1.07), ncol=2)
                plt.savefig(os.path.join(savepath, Data2Fit.columns[0] + '_' + Data2Fit.columns[1] + '.png'),
                            dpi=300, bbox_inches='tight')
                plt.show()
                j = j + 1
            else:
                plt.ylim(ymin=0)
                plt.subplots_adjust(left=0.15, bottom=0.15)
                plt.legend(loc='upper center', bbox_to_anchor=(0,1.07), ncol=2)
                plt.savefig(os.path.join(savepath, Data2Fit.columns[0] + '_' + Data2Fit.columns[1] + '.png'),
                            dpi=300, bbox_inches='tight')
                plt.show()
            # plt.close(Figure)

        # don't use colormap if age is plotted on main axes
        else:
            # sns.regplot(x=FitResults.model.exog[:,1], y=Y_Obs, ax=Axes, scatter=False)
            Axes.plot(X[:, 1], Y_Fit, color=(1, 0, 0), label='Fit')
            # Axes.fill_between(X_Obs, Sorted_CI_o, Sorted_CI_u, color=(0, 0, 0), alpha=0.1,
            #                   label=str(int(Alpha * 100)) + '% CI')
            Axes.plot(X[:, 1][Data['Gender'] == 'M'], Y_Obs[Data['Gender'] == 'M'], linestyle='none', marker='s',
                      color=(0, 0, 0), fillstyle='none', label='male')
            Axes.plot(X[:, 1][Data['Gender'] == 'F'], Y_Obs[Data['Gender'] == 'F'], linestyle='none', marker='o',
                      color=(0, 0, 0), fillstyle='none', label='female')
            Axes.annotate(r'$N$ : ' + str(N), xy=(0.05, 0.325), xycoords='axes fraction')
            Axes.annotate(r'$R^2$ : ' + str(R2), xy=(0.05, 0.25), xycoords='axes fraction')
            Axes.annotate(r'$\sigma_{est}$ : ' + str(SE), xy=(0.05, 0.175), xycoords='axes fraction')
            Axes.annotate(r'$p$ : ' + str(p), xy=(0.05, 0.1), xycoords='axes fraction')
            Axes.annotate('95% CI [' + str(CI_l) + r'$,$ ' + str(CI_r) + ']', xy=(0.05, 0.025),
                          xycoords='axes fraction')
            Axes.set_ylabel(Data.columns[2])
            Axes.set_xlabel(Data.columns[1])

            # condition used for autoscaling
            if x_axis_abbrev == autoscale_list.loc[j][0] and y_axis_abbrev == autoscale_list.loc[j][1]:
                # plt.ylim(ymin=0, ymax=round(Y_Fit.max() * 1.2, 2))
                plt.autoscale()
                plt.subplots_adjust(left=0.15, bottom=0.15)
                plt.legend(loc='lower right')
                plt.savefig(os.path.join(savepath, Data2Fit.columns[0] + '_' + Data2Fit.columns[1] + '.png'),
                            dpi=300, bbox_inches='tight')
                plt.show()
                j = j + 1
            else:
                plt.ylim(ymin=0)
                plt.subplots_adjust(left=0.15, bottom=0.15)
                plt.legend(loc='lower right')
                plt.savefig(os.path.join(savepath, Data2Fit.columns[0] + '_' + Data2Fit.columns[1] + '.png'),
                            dpi=300, bbox_inches='tight')
                plt.show()
            # plt.close(Figure)
    # if p-value greater than 0.05, no fit will be drawn & if age is contained on main axes, no colormap will be used
    else:
        if x_axis != 'Age':
            Axes.scatter(X_np[:, 1][Data['Gender'] == 'M'], Y_Obs_np[Data['Gender'] == 'M'],
                         c=list(tuple(male_age.tolist())), cmap='plasma_r', vmin=Data['Age'].min(),
                         vmax=Data['Age'].max(), label='male', marker='s')
            Axes.scatter(X_np[:, 1][Data['Gender'] == 'F'], Y_Obs_np[Data['Gender'] == 'F'],
                         c=list(tuple(female_age.tolist())), cmap='plasma_r', vmin=Data['Age'].min(),
                         vmax=Data['Age'].max(), label='female', marker='o')
            ax = plt.gca()
            PCM = ax.get_children()[0]
            plt.colorbar(PCM, ax=ax, label='Age y')
            Axes.annotate(r'$N$ : ' + str(N), xy=(0.05, 0.325), xycoords='axes fraction')
            Axes.annotate(r'$R^2$ : ' + str(R2), xy=(0.05, 0.25), xycoords='axes fraction')
            Axes.annotate(r'$\sigma_{est}$ : ' + str(SE), xy=(0.05, 0.175), xycoords='axes fraction')
            Axes.annotate(r'$p$ : ' + format(round(p, 3), '.3f'), xy=(0.05, 0.1), xycoords='axes fraction')
            Axes.annotate('95% CI [' + str(CI_l) + r'$,$ ' + str(CI_r) + ']', xy=(0.05, 0.025),
                          xycoords='axes fraction')
            Axes.set_ylabel(Data.columns[2])
            Axes.set_xlabel(Data.columns[1])

            # condition used for autoscaling
            if x_axis_abbrev == autoscale_list.loc[j][0] and y_axis_abbrev == autoscale_list.loc[j][1]:
                # plt.ylim(ymin=0, ymax=round(Y_Fit.max() * 1.2, 2))
                plt.autoscale()
                plt.subplots_adjust(left=0.15, bottom=0.15)
                plt.legend(loc='lower right')
                plt.savefig(os.path.join(savepath, Data2Fit.columns[0] + '_' + Data2Fit.columns[1] + '.png'),
                            dpi=300, bbox_inches='tight')
                plt.show()
                j = j + 1
            else:
                plt.ylim(ymin=0)
                plt.subplots_adjust(left=0.15, bottom=0.15)
                plt.legend(loc='lower right')
                plt.savefig(os.path.join(savepath, Data2Fit.columns[0] + '_' + Data2Fit.columns[1] + '.png'),
                            dpi=300, bbox_inches='tight')
                plt.show()
            # plt.close(Figure)

        # use of colormap if age is not plotted on main axes
        else:
            Axes.plot(X[:, 1][Data['Gender'] == 'M'], Y_Obs[Data['Gender'] == 'M'], linestyle='none', marker='s',
                      color=(0, 0, 0), fillstyle='none', label='male')
            Axes.plot(X[:, 1][Data['Gender'] == 'F'], Y_Obs[Data['Gender'] == 'F'], linestyle='none', marker='o',
                      color=(0, 0, 0), fillstyle='none', label='female')
            Axes.annotate(r'$N$ : ' + str(N), xy=(0.05, 0.325), xycoords='axes fraction')
            Axes.annotate(r'$R^2$ : ' + str(R2), xy=(0.05, 0.25), xycoords='axes fraction')
            Axes.annotate(r'$\sigma_{est}$ : ' + str(SE), xy=(0.05, 0.175), xycoords='axes fraction')
            Axes.annotate(r'$p$ : ' + str(p), xy=(0.05, 0.1), xycoords='axes fraction')
            Axes.annotate('95% CI [' + str(CI_l) + r'$,$ ' + str(CI_r) + ']', xy=(0.05, 0.025),
                          xycoords='axes fraction')
            Axes.set_ylabel(Data.columns[2])
            Axes.set_xlabel(Data.columns[1])

            # condition for autoscaling
            if x_axis_abbrev == autoscale_list.loc[j][0] and y_axis_abbrev == autoscale_list.loc[j][1]:
                # plt.ylim(ymin=0, ymax=round(Y_Fit.max() * 1.2, 2))
                plt.autoscale()
                plt.subplots_adjust(left=0.15, bottom=0.15)
                plt.legend(loc='lower right')
                plt.savefig(os.path.join(savepath, Data2Fit.columns[0] + '_' + Data2Fit.columns[1] + '.png'),
                            dpi=300, bbox_inches='tight')
                plt.show()
                j = j + 1
            else:
                plt.ylim(ymin=0)
                plt.subplots_adjust(left=0.15, bottom=0.15)
                plt.legend(loc='lower right')
                plt.savefig(os.path.join(savepath, Data2Fit.columns[0] + '_' + Data2Fit.columns[1] + '.png'),
                            dpi=300, bbox_inches='tight')
                plt.show()
            # plt.close(Figure)

# Put everything into growing list and convert to DataFrame that is saved as .csv file
    values = [x_axis, y_axis, p, SE, R2, N, CI_l, CI_r]
    results.append(values)
result_dir = pd.DataFrame(results, columns=['X-axis', 'Y-axis', 'p-value', '\u03C3\u2091\u209B\u209C', 'R\u00B2', 'N',
                                            'lower bound 95% CI', 'upper bound 95% CI'])
result_dir.to_csv(os.path.join(savepath, 'ResultsPlots.csv'), index=False)

# boxplots of specific component weights
MWF = df['Mineral weight fraction -']
OWF = df['Organic weight fraction -']
WWF = df['Water weight fraction -']
WF = [MWF, OWF, WWF]

fig = plt.figure(figsize=(5.5, 4.5))
ax1 = fig.add_subplot(111)
bp = ax1.boxplot(WF)
ax1.set_ylabel('Weight Fraction -')
ax1.set_xticklabels(['Mineral', 'Organic', 'Water'])
plt.ylim(ymin=0)
plt.savefig(os.path.join(savepath, 'WF_boxplt.png'), dpi=300, bbox_inches='tight')
plt.show()



# boxplot of AMM/AMD
AMM = df['Apparent Modulus Mineralized MPa'].dropna().reset_index(drop=True)
AMD = df['Apparent Modulus Demineralized MPa'].dropna().reset_index(drop=True)
AMM = AMM.values.tolist()
AMD = AMD.values.tolist()
# AMM = np.array(AMM)
# AMD = np.array(AMD)

# If we were to simply plot pts, we'd lose most of the interesting
# details due to the outliers. So let's 'break' or 'cut-out' the y-axis
# into two portions - use the top (ax) for the outliers, and the bottom
# (ax2) for the details of the majority of our data
f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

# plot the same data on both axes
ax.boxplot(AMM, positions=[1])
ax2.boxplot(AMD, positions=[2])

# zoom-in / limit the view to different portions of the data
ax.set_ylim(10000, 20000)  # outliers only
ax2.set_ylim(0, 300)  # most of the data

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.tick_params(labeltop=False, labelbottom=False)  # don't put tick labels at the top
ax2.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()
ax2.set_xticklabels(['Mineralized', 'Demineralized'])

ax.set_ylabel('Apparent Modulus MPa')
ax.yaxis.set_label_coords(-0.12, 0)

# This looks pretty good, and was fairly painless, but you can get that
# cut-out diagonal lines look with just a bit more work. The important
# thing to know here is that in axes coordinates, which are always
# between 0-1, spine endpoints are at these locations (0,0), (0,1),
# (1,0), and (1,1).  Thus, we just need to put the diagonals in the
# appropriate corners of each of our axes, and so long as we use the
# right transform and disable clipping.

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'
plt.savefig(os.path.join(savepath, 'AM_boxplt.png'), dpi=300, bbox_inches='tight')
plt.show()
