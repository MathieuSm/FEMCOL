# Import standard packages
from pathlib import Path            # Used to manage path variables in windows or linux
import numpy as np                  # Used to do arrays (matrices) computations namely
import pandas as pd                 # Used to manage data frames
import matplotlib.pyplot as plt     # Used to perform plots
from skimage import filters         # Used to perform filtering image operations (e.g. Otsu)

# Import self-written script
import ISQReader

# Create class for ISQReader script input
class ISQArguments:
    Echo = True
    BMD = False
    File = None


# Set directories
CurrentDirectory = Path.cwd()
DataDirectory = CurrentDirectory / '02_Data/00_Tests'

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
ZMid, Ymid, XMid = np.round(Scan.shape / 2).astype('int')

# Plot XY mid-plane
Size = Scan.shape[1:] / 100
Figure, Axis = plt.subplots(1,1, figsize=(Size[1], Size[0]))
Axis.imshow(Scan[ZMid, :, :], cmap='bone')
Axis.axis('off')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.show()

# Segment scan using Otsu's threshold
Threshold = filters.threshold_otsu(Scan)
BinaryScan = np.zeros(Scan.shape)
BinaryScan[Scan > Threshold] = 1

# Plot segmented image
Figure, Axis = plt.subplots(1,1, figsize=(Size[1], Size[0]))
Axis.imshow(BinaryScan[ZMid, :, :], cmap='bone')
Axis.axis('off')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.show()

# Compute bone gray values
BGV = Scan * BinaryScan
print('Mean bone gray value: ' + str(round(BGV.mean(),3)))