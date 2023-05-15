import image_reader as read
import image_processing as pro
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import json as js
import scipy.ndimage as ndimage
from matplotlib_scalebar.scalebar import ScaleBar


def project_2d(img_np):
    projection = np.zeros((img_np.shape[0], img_np.shape[1]))
    for i in np.arange(img_np.shape[0]):
        for j in np.arange(img_np.shape[1]):
            projection[i, j] = np.sum(img_np[i, j, :])
    return projection


def cross(img_np, coordinate, size=5):
    img_np[int(coordinate[0]) - size:int(coordinate[0]) + size, int(coordinate[1])] = 0
    img_np[int(coordinate[0]), int(coordinate[1]) - size:int(coordinate[1]) + size] = 0
    return img_np


def big_cross(img_np, coordinate, value=0):
    img_np[:, int(coordinate[1])] = value
    img_np[int(coordinate[0]), :] = value
    return img_np


def plot_segmentation(img_gray, img_seg, title='', alpha=0.6, save_path=''):
    img_seg = img_seg.astype(int)
    img_seg.dtype = 'bool'
    mask = np.ma.array(img_seg, mask=~img_seg)

    plt.figure()
    plt.title(title)
    plt.imshow(img_gray, cmap='Greys_r')
    plt.imshow(mask, cmap=plt.cm.flag, interpolation='none', vmin=img_gray.min(), vmax=img_gray.max(),
               alpha=alpha)
    if save_path != '':
        plt.savefig(save_path)


json_path = '../01_Data/6_Neck_paper_template/shift_list_no_torque_shorted.json'
# read.read()


with open(json_path) as f:
    data = js.load(f)
my_dpi = 600
mm = 1 / 25.4
for img_name_json in list(data):
    img_name = img_name_json.split('.ISQ')[0] + '.mhd'
    if 'C0002337' in img_name:
        print(img_name)
        img_grey_np, header_grey = read.read('../01_Data/3_2_uCT_GREY_DOWNSCALED_0_13_mm_BBCUT/' + img_name)
        img_seg_np, header_seg = read.read('../01_Data/4_2_uCT_SEG_DOWNSCALED_0_13_mm_CLEAN_BBCUT/' + img_name)
        img_mask_np, header_mask = read.read('../01_Data/5_2_uCT_MASK_DOWNSCALED_0_13_mm_BBCUT/' + img_name)

        proj = np.sum(img_grey_np, axis=2)
        # plt.figure()
        # plt.imshow(proj)
        #
        # plt.figure()
        # plt.imshow(np.rot90(img_seg_np[:, :, -2], k=2, axes=(0, 1)))

        cog = ndimage.measurements.center_of_mass(proj)

        coo_not_torque_pix = np.array(data[img_name_json]) / np.array(header_grey['resolution'])
        proj_cog = cross(proj, cog)
        proj_cog_no_torque = big_cross(proj_cog, cog + coo_not_torque_pix[:2])
        proj_cog_no_torque_seg = proj_cog_no_torque + img_seg_np[:, :, -2] * 100000

        # plt.figure()
        # plt.imshow(np.rot90(proj_cog_no_torque, k=2, axes=(0, 1)))

        # plt.figure()
        # plt.imshow(np.rot90(proj_cog_no_torque_seg, k=2, axes=(0, 1)))

        # plot_segmentation(np.rot90(proj_cog_no_torque, k=2, axes=(0, 1)),
        #                   np.rot90(img_seg_np[:, :, -2], k=2, axes=(0, 1)))

        # read.plot_stack(np.rot90(img_grey_np[:, ::-1, :], k=1, axes=(0, 1)))
        # read.plot_stack(np.rot90(img_grey_np, k=2, axes=(0, 1)))
        layers = 50
        slice_seg = img_seg_np[:, :, -2]
        slice_seg_aug = np.zeros((slice_seg.shape[0] + 2 * layers, slice_seg.shape[1] + 2 * layers))
        slice_seg_aug[layers:int(slice_seg.shape[0] + layers), layers:int(slice_seg.shape[1] + layers)] = slice_seg

        slice_grey = proj_cog_no_torque
        slice_grey_aug = np.zeros((slice_grey.shape[0] + 2 * layers, slice_grey.shape[1] + 2 * layers))
        slice_grey_aug[layers:int(slice_grey.shape[0] + layers), layers:int(slice_grey.shape[1] + layers)] = slice_grey
        slice_grey_aug_cog = cross(slice_grey_aug, cog + np.array([layers] * 2).astype(float))
        slice_grey_aug_cog_no_torque = big_cross(slice_grey_aug, cog + coo_not_torque_pix[:2] + layers,
                                                 np.max(slice_grey_aug))
        slice_grey_aug_cog_no_torque_rotated = np.rot90(slice_grey_aug_cog_no_torque, k=2, axes=(0, 1))

        # plt.figure()
        # plt.imshow(slice_grey_aug_cog_no_torque_rotated)

        # mask_rot = np.rot90(img_seg_np[:, :, -2], k=2, axes=(0, 1))
        mask_rot = np.rot90(slice_seg_aug, k=2, axes=(0, 1))
        mask_rot_int = mask_rot.astype(int)
        mask_rot_int_inv = np.ma.array(mask_rot_int, mask=mask_rot_int * (-1) + 1)
        # grey = np.rot90(proj_cog_no_torque, k=2, axes=(0, 1))
        # plt.figure(figsize=(slice_grey_aug.shape[0] / my_dpi / mm * header_grey['resolution'][0],
        #                     slice_grey_aug.shape[1] / my_dpi / mm * header_grey['resolution'][0]), dpi=my_dpi)
        # plt.figure(figsize=(3.937007, 3.937007), frameon=False)


        fig = plt.figure(frameon=False)
        fig.set_size_inches(slice_grey_aug.shape * np.array(header_grey['resolution'][:2]) / 25.4)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(slice_grey_aug_cog_no_torque_rotated, cmap='Greys_r',
                  extent=[0, slice_grey_aug.shape[0] * header_grey['resolution'][0],
                          slice_grey_aug.shape[1] * header_grey['resolution'][0], 0], aspect='auto')
        ax.imshow(mask_rot_int_inv, cmap=plt.cm.flag, interpolation='none', vmin=0, vmax=np.max(slice_grey_aug),
                  alpha=0.6, extent=[0, slice_grey_aug.shape[0] * header_grey['resolution'][0],
                                     slice_grey_aug.shape[1] * header_grey['resolution'][0], 0], aspect='auto')
        plt.gca().add_artist(ScaleBar(1, 'mm'))
        plt.tight_layout()
        plt.savefig('../01_Data/6_Neck_paper_template/' + img_name.replace('mhd', 'pdf'),
                    dpi=1 / (np.array(header_grey['resolution'][0]) / 25.4))
        print('\n')

        # slice_grey_aug.shape = (352, 344) in pixel
        # slice_grey_aug.shape * np.array(header_grey['resolution'][:2]) = array([46.18166667, 45.13208333]) in mm
        # inch = array([1.81817585, 1.77685367])
        # dpi = 193.6
