# This script loads .mhd files and calculates cross-sectional area.

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
import cv2

def cv2_open(seg_np, radius, axis=2, disp_tqdm=True):
    print('    ... start cv2_open')
    opend = np.zeros(seg_np.shape)
    kernel_2d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    if disp_tqdm:
        if axis == 0:
            for i in tqdm(np.arange(seg_np.shape[axis])):
                opend[i, :, :] = cv2.morphologyEx(seg_np[i, :, :], cv2.MORPH_OPEN, kernel_2d)
        elif axis == 1:
            for i in tqdm(np.arange(seg_np.shape[axis])):
                opend[:, i, :] = cv2.morphologyEx(seg_np[:, i, :], cv2.MORPH_OPEN, kernel_2d)
        elif axis == 2:
            for i in tqdm(np.arange(seg_np.shape[axis])):
                opend[:, :, i] = cv2.morphologyEx(seg_np[:, :, i], cv2.MORPH_OPEN, kernel_2d)
    else:
        if axis == 0:
            for i in np.arange(seg_np.shape[axis]):
                opend[i, :, :] = cv2.morphologyEx(seg_np[i, :, :], cv2.MORPH_OPEN, kernel_2d)
        elif axis == 1:
            for i in np.arange(seg_np.shape[axis]):
                opend[:, i, :] = cv2.morphologyEx(seg_np[:, i, :], cv2.MORPH_OPEN, kernel_2d)
        elif axis == 2:
            for i in np.arange(seg_np.shape[axis]):
                opend[:, :, i] = cv2.morphologyEx(seg_np[:, :, i], cv2.MORPH_OPEN, kernel_2d)
    return opend.astype('uint8')

# Set directories
CurrentDirectory = Path.cwd()
ScriptsDirectory = CurrentDirectory / 'Silk/01_Silk_scripts'
DataDirectory = CurrentDirectory / 'Silk/00_Silk_data/00_uCT'
ResultsDirectory = CurrentDirectory / 'Silk/02_Silk_results/00_uCT'

# Read data list and print it into console
Data = pd.read_csv(str(DataDirectory / 'SampleList.csv'))
Data = Data.dropna().reset_index(drop=True)
# MeanOtsu = pd.read_csv(CurrentDirectory / '04_Results/03_uCT/MeanOtsu.csv')
print(Data)
# print('Mean Otsu Threshold = ' + str(MeanOtsu.loc[0][0]))

results = list()
values = list()

# Select sample to analyze (numbering starting from 0)
for x in range(0, 2):
    SampleNumber = x

    # SampleID = Data.loc[SampleNumber, 'Sample']
    File = Data.loc[SampleNumber, 'uCT File']

    # Read mhd file
    Image_f = sitk.ReadImage(str(DataDirectory / File) + '_free' + '.mhd')
    Image_s = sitk.ReadImage(str(DataDirectory / File) + '_stretched' + '.mhd')
    Scan_f = sitk.GetArrayFromImage(Image_f)
    Scan_s = sitk.GetArrayFromImage(Image_s)

    # Compute scan mid-planes positions
    ZMid_f, YMid_f, XMid_f = np.round(np.array(Scan_f.shape) / 2).astype('int')
    ZMid_s, YMid_s, XMid_s = np.round(np.array(Scan_s.shape) / 2).astype('int')

    # Plot XY mid-plane
    Size_f = np.array(Scan_f.shape[1:]) / 100
    Figure, Axis = plt.subplots(1, 1, figsize=(Size_f[1], Size_f[0]))
    Axis.imshow(Scan_f[ZMid_f, :, :], cmap='Greys_r')
    Axis.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.show()
    # plt.close()

    # Plot YZ mid-plane
    Figure, Axis = plt.subplots(1, 1, figsize=(Size_f[1], Size_f[0]))
    Axis.imshow(Scan_f[:, :, XMid_f], cmap='Greys_r')
    Axis.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    # plt.savefig(os.path.join('/home/stefan/Documents/FEMCOL/Silk_data/', 'x' + '_' + 'YZ_Plane'),
    #             dpi=300)
    # plt.show()
    plt.close()

    # Plot XY mid-plane
    Size_s = np.array(Scan_s.shape[1:]) / 100
    Figure, Axis = plt.subplots(1, 1, figsize=(Size_s[1], Size_s[0]))
    Axis.imshow(Scan_s[ZMid_s, :, :], cmap='Greys_r')
    Axis.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.show()

    # Plot YZ mid-plane
    Figure, Axis = plt.subplots(1, 1, figsize=(Size_s[1], Size_s[0]))
    Axis.imshow(Scan_s[:, :, XMid_s], cmap='Greys_r')
    Axis.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    # plt.savefig(os.path.join('/home/stefan/Documents/FEMCOL/Silk_data/', 'x' + '_' + 'YZ_Plane'),
    #             dpi=300)
    # plt.show()
    plt.close()

    # Segment scan using Otsu's threshold
    Threshold_f = filters.threshold_otsu(Scan_f)
    Threshold_s = filters.threshold_otsu(Scan_s)
    BinaryScan_f = np.zeros(Scan_f.shape)
    BinaryScan_s = np.zeros(Scan_s.shape)
    BinaryScan_f[Scan_f > Threshold_f] = 1
    BinaryScan_s[Scan_s > Threshold_s] = 1

    # Plot segmented image
    Figure, Axis = plt.subplots(1,1, figsize=(Size_f[1], Size_f[0]))
    Axis.imshow(BinaryScan_f[ZMid_f, :, :], cmap='Greys_r')
    Axis.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.show()
    # plt.close()

    # Plot segmented image
    Figure, Axis = plt.subplots(1,1, figsize=(Size_s[1], Size_s[0]))
    Axis.imshow(BinaryScan_s[ZMid_s, :, :], cmap='Greys_r')
    Axis.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.show()
    # plt.close()

    # Pad image to avoid artefacts
    Radius = 5
    Pad = int(Radius)
    Padded_f = np.pad(BinaryScan_f, Pad)
    Padded_s = np.pad(BinaryScan_s, Pad)
    print(Padded_f[ZMid_f, :, :].sum())
    print(Padded_s[ZMid_s, :, :].sum())

    Figure, (Axis1, Axis2) = plt.subplots(1, 2, figsize=(2*Size_f[1], Size_f[0]))
    Axis1.imshow(Padded_f[ZMid_f, :, :], cmap='Greys_r')
    Axis1.axis('off')
    Axis2.imshow(Padded_s[ZMid_s, :, :], cmap='Greys_r')
    Axis2.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.show()
    # # plt.close()
    #
    # Fill pore to estimate surface
    # Disk = morphology.disk(Radius)
    # Dilated_f = morphology.binary_dilation(Padded_f[ZMid_f+int(Radius), :, :], Disk)
    # Dilated_s = morphology.binary_dilation(Padded_s[ZMid_s+int(Radius), :, :], Disk)
    # Eroded_f = morphology.binary_erosion(Dilated_f, Disk)[Pad:-Pad, Pad:-Pad]
    # Eroded_s = morphology.binary_erosion(Dilated_s, Disk)[Pad:-Pad, Pad:-Pad]
    #
    # Figure, Axis = plt.subplots(1, 1, figsize=(Size_f[1], Size_f[0]))
    # Axis.imshow(Eroded_f, cmap='Greys_r')
    # Axis.axis('off')
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    # plt.show()
    # # plt.close()
    #
    # Figure, Axis = plt.subplots(1, 1, figsize=(Size_s[1], Size_s[0]))
    # Axis.imshow(Eroded_s, cmap='Greys_r')
    # Axis.axis('off')
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    # plt.show()
    # # plt.close()

    # cut off upper connection and show binarized image
    if SampleNumber == 1:
        for j in range(178, 220):
            for k in range(0, 350):
                BinaryScan_f[ZMid_f, k, j] = 0

        for j in range(200, 220):
            for k in range(340, 500):
                BinaryScan_f[ZMid_f, k, j] = 0

        image_f_opened = cv2_open(BinaryScan_f, 15, axis=0, disp_tqdm=True)
        Figure, (Axis1, Axis2, Axis3) = plt.subplots(1, 3, figsize=(3 * Size_f[1], Size_f[0]))
        Axis1.imshow(Scan_f[ZMid_f, :, :], cmap='Greys_r')
        Axis1.axis('off')
        Axis1.set_title('Scan')
        Axis3.imshow(image_f_opened[ZMid_f, :, :], cmap='Greys_r')
        Axis3.axis('off')
        Axis3.set_title('Opened')
        Axis2.imshow(BinaryScan_f[ZMid_f, :, :], cmap='Greys_r')
        Axis2.axis('off')
        Axis2.set_title('Binarized')
        # Axis4.imshow(Eroded_f, cmap='Greys_r')
        # Axis4.axis('off')
        # Axis4.set_title('Eroded')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        plt.savefig(os.path.join(ResultsDirectory, 'SilkImageProcessingStepsDry.png'), dpi=300, bbox_inches='tight',
                    format='png')
        plt.show()

        Voxel_Dimensions = np.array(Image_f.GetSpacing())
        Voxel_Area = Voxel_Dimensions[0] * Voxel_Dimensions[1]
        cross_section_f = image_f_opened[ZMid_f, :, :].sum() * Voxel_Area
        cross_section_s = BinaryScan_s[ZMid_s, :, :].sum() * Voxel_Area

        remark = 'dry'

    else:
        for j in range(250, 260):
            for k in range(0, 250):
                BinaryScan_f[ZMid_f, k, j] = 0

        Figure, (Axis1, Axis2) = plt.subplots(1, 2, figsize=(3*Size_f[1], Size_f[0]))
        Axis1.imshow(Scan_f[ZMid_f, :, :], cmap='Greys_r')
        Axis1.axis('off')
        Axis1.set_title('Scan')
        Axis2.imshow(BinaryScan_f[ZMid_f, :, :], cmap='Greys_r')
        Axis2.axis('off')
        Axis2.set_title('Binarized')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        plt.savefig(os.path.join(ResultsDirectory, 'SilkImageProcessingStepsWet.png'), dpi=300, bbox_inches='tight', format='png')
        plt.show()

        Voxel_Dimensions = np.array(Image_f.GetSpacing())
        Voxel_Area = Voxel_Dimensions[0] * Voxel_Dimensions[1]
        cross_section_f = BinaryScan_f[ZMid_f, :, :].sum() * Voxel_Area
        cross_section_s = BinaryScan_s[ZMid_s, :, :].sum() * Voxel_Area

        remark = 'wet'

        # Collect data into filling list
    values = [SampleNumber, remark, round(cross_section_s, 2), round(cross_section_f, 2)]
    results.append(values)

result_dir = pd.DataFrame(results, columns=['Sample Number', 'Sample State', 'Area of pre-stretched strap / mm^2',
                                            'Area of free strap / mm^2'])
result_dir.to_csv(os.path.join(ResultsDirectory, 'CrossSectionalArea.csv'), index=False)
print(result_dir)

