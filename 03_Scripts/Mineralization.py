# This script loads .mhd files and calculates various quantities such as BMC, BMD, TMD, BV/TV, etc.

# Import standard packages
from pathlib import Path                 # Used to manage path variables in windows or linux
import numpy as np                       # Used to do arrays (matrices) computations namely
import pandas as pd                      # Used to manage data frames
import SimpleITK as sitk                 # Used to read images
import matplotlib.pyplot as plt          # Used to perform plots
import os
import math
import statistics
from skimage import filters              # Used to perform filtering image operations (e.g. Otsu)
from skimage import morphology, measure  # Used to fill pores and circle fitting
from tqdm import tqdm

# Set directories
CurrentDirectory = Path.cwd()
ScriptsDirectory = CurrentDirectory / '03_Scripts'
DataDirectory = CurrentDirectory / '02_Data/01_uCT'

# Read data list and print it into console
Data = pd.read_csv(str(DataDirectory / 'SampleList.csv'))
Data = Data.drop('Remark', axis=1)
Data = Data.dropna().reset_index(drop=True)
MeanOtsu = pd.read_csv(CurrentDirectory / '04_Results/03_uCT/MeanOtsu.csv')
print(Data)
print('Mean Otsu Threshold = ' + str(MeanOtsu.loc[0][0]))

results = list()
Pi = 3.14159265

# Select sample to analyze (numbering starting from 0)
for x in tqdm(range(0, len(Data), 1)):
    SampleNumber = x
    SampleID = Data.loc[SampleNumber, 'Sample']
    File = Data.loc[SampleNumber, 'uCT File']

    # Read mhd file
    Image = sitk.ReadImage(str(DataDirectory / File) + '.mhd')
    Scan = sitk.GetArrayFromImage(Image)

    # Compute scan mid-planes positions
    ZMid, YMid, XMid = np.round(np.array(Scan.shape) / 2).astype('int')

    # Plot XY mid-plane
    Size = np.array(Scan.shape[1:]) / 100
    Figure, Axis = plt.subplots(1, 1, figsize=(Size[1], Size[0]))
    Axis.imshow(Scan[ZMid, :, :], cmap='bone')
    Axis.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    # plt.show()
    plt.close()

    # Plot YZ mid-plane
    Figure, Axis = plt.subplots(1, 1, figsize=(Size[1], Size[0]))
    Axis.imshow(Scan[:, :, YMid], cmap='bone')
    Axis.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.savefig(os.path.join('/home/stefan/PycharmProjects/FEMCOL/04_Results/03_uCT/', SampleID + '_' + 'YZ_Plane'),
                dpi=300)
    # plt.show()
    plt.close()

    # Segment scan using Otsu's threshold: mean threshold calculated with MeanOtsu.py --> 562.586
    Threshold = MeanOtsu.loc[0][0]
    BinaryScan = np.zeros(Scan.shape)
    BinaryScan[Scan > Threshold] = 1
    SegmentedImage = sitk.GetImageFromArray(BinaryScan)
    SegmentedImage.SetSpacing(Image.GetSpacing())
    SegmentedImage.SetOrigin(Image.GetOrigin())
    SegmentedImage.SetDirection(Image.GetDirection())
    sitk.WriteImage(SegmentedImage, '/home/stefan/PycharmProjects/FEMCOL/04_Results/03_uCT/SegmentedImage_' + SampleID + '.mhd')

    # Plot segmented image
    Figure, Axis = plt.subplots(1,1, figsize=(Size[1], Size[0]))
    Axis.imshow(BinaryScan[ZMid, :, :], cmap='bone')
    Axis.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    # plt.show()
    plt.close()

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
    # plt.show()
    plt.close()

    # Build full cylinder representing the sample
    Disk = morphology.disk(int(round(RegionProperties.equivalent_diameter/2)))
    Ceil = np.ceil((np.array(Scan.shape[1:]) - np.array(Disk.shape)) / 2).astype('int')
    Floor = np.floor((np.array(Scan.shape[1:]) - np.array(Disk.shape)) / 2).astype('int')
    Shift = np.array(Scan.shape[1:])/2 - np.array([Y0, X0])

    Padded = np.pad(Disk, ((Floor[1]-int(Shift[1]), Ceil[1]+int(Shift[1])), (Floor[0]-int(Shift[0]),
                                                                             Ceil[0]+int(Shift[0]))))
    Cylinder = np.repeat(Padded, Scan.shape[0]).reshape(Scan.shape, order='F')

    Figure, Axis = plt.subplots(1, 1, figsize=(Size[1], Size[0]))
    Axis.imshow(Cylinder[ZMid, :, :], cmap='bone')
    Axis.plot(X0, Y0, marker='x', color=(0, 0, 1), linestyle='none', markersize=10, mew=2, label='Centroid')
    Axis.plot(X0 + RotatedEllipse[0, :], Y0 - RotatedEllipse[1, :], color=(0, 1, 0), label='Fitted ellipse')
    Axis.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    # plt.show()
    plt.close()

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
    # plt.show()
    plt.close()

    BMD = round(Sample.sum() / Cylinder.sum(), 3)
    TMD = round(Tissue.sum() / BinaryScan.sum(), 3)

    # print('Mean bone mineral density: ' + str(BMD) + ' mg HA / cm3')
    # print('Mean tissue mineral density: ' + str(TMD) + ' mg HA / cm3')

    Voxel_Dimensions = np.array(Image.GetSpacing()) * 10**-3
    Voxel_Volume = Voxel_Dimensions[0] * Voxel_Dimensions[1] * Voxel_Dimensions[2]

    BMC = round(Tissue.sum() * BinaryScan.sum() * Voxel_Volume, 3)
    # print('Bone mineral content: ' + str(round(BMC, 3)) + ' mg HA')

    # Calculate area of segmented image (considering porosity), apparent area (total area without considering porosity),
    # and area fraction (bone area/total area)
    Area = Voxel_Dimensions[0] * Voxel_Dimensions[1]
    BoneAreas = np.zeros(Scan.shape[0])
    BoneAreas_new = np.zeros(Scan.shape[0])
    TotalAreas = np.zeros(Scan.shape[0])
    Areas_fraction = np.zeros(Scan.shape[0])
    Areas_fraction_new = np.zeros(Scan.shape[0])
    BoneVolumes = np.zeros(Scan.shape[0])

    for i in range(Scan.shape[0]):
        BoneAreas[i] = BinaryScan[i].sum() * Area * 1e06
        BoneAreas_new[i] = Sample[i].sum() * Area * 1e03
        TotalAreas[i] = Cylinder[i].sum() * Area * 1e06
        Areas_fraction[i] = BoneAreas[i] / TotalAreas[i]
        Areas_fraction_new[i] = BoneAreas_new[i] / TotalAreas[i]
        # BoneVolumes[i] = BoneAreas[i] * Voxel_Dimensions[2] * 1e03
    min_BoneArea_wp = round(BoneAreas.min(), 3)

    stderr_BoneArea_wp = round(statistics.stdev(BoneAreas), 8)
    mean_BoneArea_wp = round(statistics.mean(BoneAreas), 3)
    coeff_var_BoneArea_wp = round(stderr_BoneArea_wp/mean_BoneArea_wp, 3)

    mean_Area_wop = round(statistics.mean(TotalAreas), 3)
    min_Diam_wp = round(math.sqrt(min_BoneArea_wp/Pi*4), 3)
    mean_Diam_wop = round(math.sqrt(mean_Area_wop/Pi*4), 3)
    min_areas_fraction = round(Areas_fraction.min(), 3)
    min_areas_fraction_new = round(Areas_fraction_new.min(), 3)
    mean_areas_fraction = round(statistics.mean(Areas_fraction), 3)

    # TotalVolume_mean = mean_Area_wop * 946 * Voxel_Dimensions[0] * 1e03
    # TotalVolume_filled = RegionProperties.area
    # BVTV_new = round(BoneVolumes.sum() / TotalVolume_mean, 3)

    # Collect data into filling list
    values = [SampleID, BVTV, BMD, TMD, BMC, min_BoneArea_wp, min_Diam_wp, mean_Area_wop, mean_Diam_wop,
              mean_areas_fraction, min_areas_fraction, min_areas_fraction_new, coeff_var_BoneArea_wp, mean_BoneArea_wp]
    results.append(values)

# Add missing samples
missing_sample_IDs = pd.DataFrame({'Sample ID': ['390_R', '395_R', '402_L']})

# convert list to dataframe
result_dir = pd.DataFrame(results, columns=['Sample ID', 'Bone Volume Fraction / -', 'Bone Mineral Density mg HA / cm3',
                                            'Tissue Mineral Density mg HA / cm^3', 'Bone Mineral Content / mg HA',
                                            'Min ECM Area / mm^2', 'Min Diameter / mm', 'Mean Apparent Area / mm^2',
                                            'Mean Apparent Diameter / mm', 'Mean Area Fraction / -',
                                            'Min Area Fraction / -', 'Min Area Fraction Adjusted / -',
                                            'Coefficient of Variation / -', 'Mean ECM Area / mm^2'])
result_dir = pd.concat([result_dir, missing_sample_IDs])
result_dir_sorted = result_dir.sort_values(by=['Sample ID'], ascending=True)
result_dir_sorted = result_dir_sorted.reset_index(drop=True)
result_dir_sorted.to_csv(os.path.join('/home/stefan/PycharmProjects/FEMCOL/04_Results/03_uCT/', 'ResultsUCT.csv'),
                         index=False)
print(result_dir_sorted)
