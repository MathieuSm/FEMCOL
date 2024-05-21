# This script is used to calculate a segmentation threshold using Otsu's method

# Import standard packages
import pandas as pd
import SimpleITK as sitk
from skimage import filters
import statistics
import os
from tqdm import tqdm


# Set directories
Cwd = os.getcwd()
DataDirectory = str(os.path.dirname(Cwd) + '/02_Data/01_uCT')
ResultsDirectory = str(os.path.dirname(Cwd) + '/04_Results/03_uCT/')

# Read data list and print it into console
Data = pd.read_csv(str(DataDirectory + '/SampleList.csv'))
Data = Data.drop('Remark', axis=1)
Data = Data.dropna().reset_index(drop=True)
print(Data)

ThresholdValues = list()

# Select sample to analyze (numbering starting from 0)
for x in tqdm(range(0, len(Data), 1)):
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

# Create dataframe with threshold values, take mean and put into new dataframe which is saved as .csv
result_dir = pd.DataFrame(ThresholdValues, columns=['Sample ID', 'Otsu Threshold'])
mean_otsu = list()
mean_otsu.append(round(statistics.mean(result_dir['Otsu Threshold']), 3))
mean_otsu = pd.DataFrame(mean_otsu, columns=['Mean Otsu Threshold'])

mean_otsu.to_csv(os.path.join(ResultsDirectory, 'MeanOtsu.csv'), index=False)
