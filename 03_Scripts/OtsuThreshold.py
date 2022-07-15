# Import standard packages
from pathlib import Path                # Used to manage path variables in windows or linux
import numpy as np                      # Used to do arrays (matrices) computations namely
import pandas as pd                     # Used to manage data frames
import SimpleITK as sitk                # Used to read images
import matplotlib.pyplot as plt         # Used to perform plots
from skimage import filters             # Used to perform filtering image operations (e.g. Otsu)
from skimage import morphology, measure # Used to fill pores and circle fitting
import statistics                       # Used to perform statistics



# Set directories
CurrentDirectory = Path.cwd()
ScriptsDirectory = CurrentDirectory / '03_Scripts'
DataDirectory = CurrentDirectory / '02_Data/01_uCT'


# Read data list and print it into console
Data = pd.read_csv(str(DataDirectory / 'SampleList.csv'))
Data = Data.drop('Remark', axis=1)
Data = Data.dropna().reset_index(drop=True)
print(Data)

ThresholdValues = list()

# Select sample to analyze (numbering starting from 0)
for x in range(0, 36, 1):
    SampleNumber = x
    File = Data.loc[SampleNumber,'uCT File']

    # Read mhd file
    Image = sitk.ReadImage(str(DataDirectory / File) + '.mhd')
    Scan = sitk.GetArrayFromImage(Image)

    # Compute scan mid-planes positions
    ZMid, Ymid, XMid = np.round(np.array(Scan.shape) / 2).astype('int')

    # Plot XY mid-plane
    Size = np.array(Scan.shape[1:]) / 100
    Figure, Axis = plt.subplots(1,1, figsize=(Size[1], Size[0]))
    Axis.imshow(Scan[ZMid, :, :], cmap='bone')
    Axis.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.close()
    # plt.show()

    # Segment scan using Otsu's threshold
    Threshold = filters.threshold_otsu(Scan)
    BinaryScan = np.zeros(Scan.shape)
    BinaryScan[Scan > Threshold] = 1

    SampleID = Data.loc[SampleNumber, 'Sample']
    values = [SampleID, Threshold]
    ThresholdValues.append(values)
    print(x)

result_dir = pd.DataFrame(ThresholdValues, columns=['Sample ID', 'Otsu Threshold'])
mean_otsu = round(statistics.mean(result_dir['Otsu Threshold']),3)
print(mean_otsu)
