# Import standard packages
from pathlib import Path                # Used to manage path variables in windows or linux
import numpy as np                      # Used to do arrays (matrices) computations namely
import pandas as pd                     # Used to manage data frames
import SimpleITK as sitk                # Used to read images
import matplotlib.pyplot as plt         # Used to perform plots
import os
from skimage import filters             # Used to perform filtering image operations (e.g. Otsu)
from skimage import morphology, measure # Used to fill pores and circle fitting

# Set directories
CurrentDirectory = Path.cwd()
ScriptsDirectory = CurrentDirectory / '03_Scripts'
DataDirectory = CurrentDirectory / '02_Data/01_uCT'

# Read data list and print it into console
Data = pd.read_csv(str(DataDirectory / 'SampleList.csv'))
Data = Data.drop('Remark', axis=1)
Data = Data.dropna().reset_index(drop=True)
print(Data)

results = list()
# Select sample to analyze (numbering starting from 0)
for x in range(0, len(Data)-1, 1):
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
    plt.show()

    # Segment scan using Otsu's threshold: mean threshold calculated with MeanOtsu.py --> 562.586
    Threshold = 562.586
    BinaryScan = np.zeros(Scan.shape)
    BinaryScan[Scan > Threshold] = 1

    # Plot segmented image
    Figure, Axis = plt.subplots(1,1, figsize=(Size[1], Size[0]))
    Axis.imshow(BinaryScan[ZMid, :, :], cmap='bone')
    Axis.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.show()

    # Pad image to avoid artefacts
    Radius = 50
    Pad = int(Radius)
    Padded = np.pad(BinaryScan, Pad)

    # Fill pore to estimate surface and compute BV/TV
    Disk = morphology.disk(Radius)
    Dilated = morphology.binary_dilation(Padded[ZMid+int(Radius), :, :], Disk)
    Eroded = morphology.binary_erosion(Dilated, Disk)[Pad:-Pad, Pad:-Pad]

    RegionProperties = measure.regionprops(Eroded*1)[0]

    # Check that properties are correctly measured
    Y0, X0 = RegionProperties.centroid
    R1 = RegionProperties.major_axis_length * 0.5
    R2 = RegionProperties.minor_axis_length * 0.5
    OrientationAngle = RegionProperties.orientation

    Radians = np.linspace(0, 2 * np.pi, 100)
    Ellipse = np.array([R1 * np.cos(Radians), R2 * np.sin(Radians)])
    R = np.array([[np.cos(OrientationAngle), -np.sin(OrientationAngle)],
                  [np.sin(OrientationAngle), np.cos(OrientationAngle)]])
    RotatedEllipse = np.dot(R, Ellipse)

    # Plot filled image
    Figure, Axis = plt.subplots(1, 1, figsize=(Size[1], Size[0]))
    Axis.imshow(Eroded, cmap='bone')
    Axis.plot(X0, Y0, marker='x', color=(0, 0, 1), linestyle='none', markersize=10, mew=2, label='Centroid')
    Axis.plot(X0 + RotatedEllipse[0, :], Y0 - RotatedEllipse[1, :], color=(0, 1, 0), label='Fitted ellipse')
    Axis.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.show()

    # Build full cylinder representing the sample
    Disk = morphology.disk(int(round(RegionProperties.equivalent_diameter/2)))
    Ceil = np.ceil((np.array(Scan.shape[1:]) - np.array(Disk.shape)) / 2).astype('int')
    Floor = np.floor((np.array(Scan.shape[1:]) - np.array(Disk.shape)) / 2).astype('int')
    Shift = np.array(Scan.shape[1:])/2 - np.array([Y0, X0])

    Padded = np.pad(Disk, ((Floor[1]-int(Shift[1]), Ceil[1]+int(Shift[1])), (Floor[0]-int(Shift[0]), Ceil[0]+int(Shift[0]))))
    Cylinder = np.repeat(Padded, Scan.shape[0]).reshape(Scan.shape, order='F')

    Figure, Axis = plt.subplots(1, 1, figsize=(Size[1], Size[0]))
    Axis.imshow(Cylinder[ZMid, :, :], cmap='bone')
    Axis.plot(X0, Y0, marker='x', color=(0, 0, 1), linestyle='none', markersize=10, mew=2, label='Centroid')
    Axis.plot(X0 + RotatedEllipse[0, :], Y0 - RotatedEllipse[1, :], color=(0, 1, 0), label='Fitted ellipse')
    Axis.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.show()

    # Compute BV/TV
    BVTV = round(BinaryScan.sum() / Cylinder.sum(), 3)
    # print('Bone volume fraction: ' + str(round(BVTV, 3)))

    # Compute bone mineral density and bone mineral content
    Sample = Scan * Cylinder
    Tissue = Scan * BinaryScan

    Figure, Axis = plt.subplots(1,1, figsize=(Size[1], Size[0]))
    Axis.imshow(Sample[ZMid, :, :], cmap='bone')
    Axis.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.show()

    BMD = round(Sample.sum() / Cylinder.sum(), 3)
    TMD = round(Tissue.sum() / BinaryScan.sum(), 3)

    # print('Mean bone mineral density: ' + str(BMD) + ' mg HA / cm3')
    # print('Mean tissue mineral density: ' + str(TMD) + ' mg HA / cm3')

    Voxel_Dimensions = np.array(Image.GetSpacing()) * 10**-3
    Voxel_Volume = Voxel_Dimensions[0] * Voxel_Dimensions[1] * Voxel_Dimensions[2]

    BMC = round(Tissue.sum() * BinaryScan.sum() * Voxel_Volume, 3)
    # print('Bone mineral content: ' + str(round(BMC, 3)) + ' mg HA')

    # Collect data into filling list
    SampleID = Data.loc[SampleNumber, 'Sample']
    values = [SampleID, BVTV, BMD, TMD, BMC]
    results.append(values)

# convert list to dataframe & save it as csv file
result_dir = pd.DataFrame(results, columns=['Sample ID', 'Bone Volume Fraction -', 'Bone Mineral Density mg HA / cm3',
                                            'Tissue Mineral Density mg HA / cm3', 'Bone Mineral Content mg HA'])
result_dir.to_csv(os.path.join('/home/stefan/Documents/PythonScripts/04_Results/03_uCT/', 'ResultsUCT.csv'), index=False)
print(result_dir)
