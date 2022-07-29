# Import standard packages
from pathlib import Path                # Used to manage path variables in windows or linux
import numpy as np                      # Used to do arrays (matrices) computations namely
import pandas as pd                     # Used to manage data frames
import SimpleITK as sitk                # Used to read images
import matplotlib.pyplot as plt         # Used to perform plots
from skimage import filters             # Used to perform filtering image operations (e.g. Otsu)
from skimage import morphology, measure # Used to fill pores and circle fitting
import statistics                       # Used to perform statistics
import os


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
    File = Data.loc[SampleNumber, 'uCT File']

    # Read mhd file
    Image = sitk.ReadImage(str(DataDirectory / File) + '.mhd')
    Scan = sitk.GetArrayFromImage(Image)

    # Segment scan using Otsu's threshold
    Threshold = filters.threshold_otsu(Scan)

    # Produce growing list with sample ID and Otsu threshold value
    SampleID = Data.loc[SampleNumber, 'Sample']
    values = [SampleID, Threshold]
    ThresholdValues.append(values)
    print(x)

# Create dataframe with threshold values, take mean and put into new dataframe which is saved as .csv
result_dir = pd.DataFrame(ThresholdValues, columns=['Sample ID', 'Otsu Threshold'])
mean_otsu = list()
mean_otsu.append(round(statistics.mean(result_dir['Otsu Threshold']), 3))
mean_otsu = pd.DataFrame(mean_otsu, columns=['Mean Otsu Threshold'])

mean_otsu.to_csv(os.path.join('/home/stefan/Documents/PythonScripts/04_Results/03_uCT/', 'MeanOtsu.csv'), index=False)
