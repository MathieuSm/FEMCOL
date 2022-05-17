# Import standard packages
from pathlib import Path                # Used to manage path variables in windows or linux
import numpy as np                      # Used to do arrays (matrices) computations namely
import pandas as pd                     # Used to manage data frames
import matplotlib.pyplot as plt         # Used to perform plots
from skimage import filters             # Used to perform filtering image operations (e.g. Otsu)
import statsmodels.formula.api as smf   # Used for statistical analysis (ols here)
from scipy.stats.distributions import t # Used to compute confidence intervals


# Set directories
CurrentDirectory = Path.cwd()
ScriptsDirectory = CurrentDirectory / '03_Scripts'
DataDirectory = CurrentDirectory / '02_Data/00_Tests'

# Import self-written script
import sys
sys.path.append(str(ScriptsDirectory))
import ISQReader

# Create class for ISQReader script input
class ISQArguments:
    Echo = True
    BMD = False
    File = None

# Define some functions
def PlotRegressionResults(Model,Alpha=0.95):

    print(Model.summary())

    ## Plot results
    Y_Obs = Model.model.endog
    Y_Fit = Model.fittedvalues
    N = int(Model.nobs)
    C = np.matrix(Model.cov_params())
    X = np.matrix(Model.model.exog)
    X_Obs = np.sort(np.array(X[:,1]).reshape(len(X)))


    ## Compute R2 and standard error of the estimate
    E = Y_Obs - Y_Fit
    RSS = np.sum(E ** 2)
    SE = np.sqrt(RSS / Model.df_resid)
    TSS = np.sum((Model.model.endog - Model.model.endog.mean()) ** 2)
    RegSS = TSS - RSS
    R2 = RegSS / TSS

    ## Compute CI lines
    B_0 = np.sqrt(np.diag(np.abs(X * C * X.T)))
    t_Alpha = t.interval(Alpha, N - X.shape[1] - 1)
    CI_Line_u = Y_Fit + t_Alpha[0] * SE * B_0
    CI_Line_o = Y_Fit + t_Alpha[1] * SE * B_0


    ## Plots
    DPI = 100
    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=DPI, sharey=True, sharex=True)
    Axes.plot(X[:,1], Y_Fit, color=(1,0,0), label='Fit')
    Axes.fill_between(X_Obs, np.sort(CI_Line_o), np.sort(CI_Line_u), color=(0, 0, 0), alpha=0.1, label=str(Alpha*100) + '% CI')
    Axes.plot(X[:,1], Y_Obs, linestyle='none', marker='o', color=(0,0,0), fillstyle='none')
    Axes.annotate(r'$N$  : ' + str(N), xy=(0.8, 0.175), xycoords='axes fraction')
    Axes.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.8, 0.1), xycoords='axes fraction')
    Axes.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.8, 0.025), xycoords='axes fraction')
    Axes.set_ylabel('Mineral Density (mg HA/cm$^3$)')
    Axes.set_xlabel('Gray Value (-)')
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.legend()
    plt.show()
    plt.close(Figure)

# Read data list and print it into console
Data = pd.read_csv(str(DataDirectory / 'List.csv'))
print(Data)

# Select sample to analyze (numbering starting from 0)
SampleNumber = 0
File = Data.loc[SampleNumber,'uCT File']

# Read ISQ file
ISQArguments.File = str(DataDirectory / File) + '.ISQ'
FileData = ISQReader.Main(ISQArguments)

# Compute scan mid-planes positions
Scan = FileData[0]
ZMid, Ymid, XMid = np.round(np.array(Scan.shape) / 2).astype('int')

# Plot XY mid-plane
Size = np.array(Scan.shape[1:]) / 100
Figure, Axis = plt.subplots(1,1, figsize=(Size[1], Size[0]))
Axis.imshow(Scan[ZMid, :, :], cmap='bone')
plt.show()

# Crop image to desired size
Cropped = Scan[:,875:1275,800:1200]
Size = np.array(Cropped.shape[1:]) / 100
Figure, Axis = plt.subplots(1,1, figsize=(Size[1], Size[0]))
Axis.imshow(Cropped[ZMid, :, :], cmap='bone')
Axis.axis('off')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.show()

# Segment scan using Otsu's threshold
Threshold = filters.threshold_otsu(Cropped)
BinaryScan = np.zeros(Cropped.shape)
BinaryScan[Cropped > Threshold] = 1

# Plot segmented image
Figure, Axis = plt.subplots(1,1, figsize=(Size[1], Size[0]))
Axis.imshow(BinaryScan[ZMid, :, :], cmap='bone')
Axis.axis('off')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.show()

# Open QC scan and plot it
ISQArguments.File = str(DataDirectory / 'QC.ISQ')
QCData = ISQReader.Main(ISQArguments)
QCScan = QCData[0]

Size = np.array(QCScan.shape[1:]) / 100
Figure, Axis = plt.subplots(1,1, figsize=(Size[1], Size[0]))
Axis.imshow(QCScan[0, :, :], cmap='bone')
plt.show()

# Extract gray values of the 4 rods
R1 = QCScan[:,370:470,225:325]
R2 = QCScan[:,650:750,300:400]
R3 = QCScan[:,650:750,600:700]
R4 = QCScan[:,370:470,700:800]

# Built data frame with mean values and corresponding mineral densities (see pdf)
Data2Fit = pd.DataFrame({'GV': [R1.mean(), R2.mean(), R3.mean(), R4.mean()],
                         'MD': [783.9946, 410.7507, 211.7785, 100.3067]},
                        index=['R1','R2','R3','R4'])

FitResults = smf.ols('MD ~ 1 + GV', data=Data2Fit).fit()
PlotRegressionResults(FitResults)


# Compute bone mineral density and bone mineral content
BMDs = FitResults.params['Intercept'] + (Cropped * BinaryScan) * FitResults.params['GV']
print('Mean bone mineral density: ' + str(round(BMDs.mean(),3)) + ' mg HA / cm3')

Voxel_Dimensions = np.array(FileData[1]['ElementSpacing']) * 10**-3
Voxel_Volume = Voxel_Dimensions[0] * Voxel_Dimensions[1] * Voxel_Dimensions[2]
VoxelNumber = BinaryScan.sum()

BMC = BMDs.sum() * VoxelNumber * Voxel_Volume
print('Bone mineral content: ' + str(round(BMC,3)) + ' mg HA')
