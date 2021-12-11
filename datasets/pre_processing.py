import numpy as np
import cv2
from skimage.morphology import skeletonize
import os
import glob
import shutil

def rgb_standarization(img, axis=(0,1)):
    mean = np.mean(img, axis=axis, keepdims=True)
    std = np.sqrt(((img - mean)**2).mean(axis=axis, keepdims=True))
    out = (img - mean) / (std + 1e-10)
    return out
    

def gray_norm(img):
    return (img-np.mean(img)) / (np.std(img)+1e-10)


def contrast_stretch(img):
    I_strech = gray_norm(img)
    I_strech = (I_strech-np.amin(I_strech))/(np.amax(I_strech)-np.amin(I_strech)+1e-10)
    if img.dtype == np.uint8:
        return (I_strech*255).astype(np.uint8)
    elif img.dtype == np.uint16 or img.dtype == np.float32:
        return (I_strech*65535).astype(np.uint16)


def hist_equalization(img):
    colors = len(img.shape)
    if colors == 2:
        img_equalized = cv2.equalizeHist(img)
    else:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        img_yuv[...,0] = cv2.equalizeHist(img_yuv[...,0])
        img_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YCR_CB2BGR)
    return img_equalized



def adjust_gamma(img, gamma=1.2, mode='uint8'):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    new_img = cv2.LUT(img, table)
    return new_img


if __name__ == "__main__":

    # img_dir = '/data/ldap_shared/home/s_lyr/code/noisyLabel/datasets/nuclei/train/masks'
    # save_dir = '/data/ldap_shared/home/s_lyr/code/noisyLabel/datasets/nuclei/train/skl'
    # img_list = glob.glob(os.path.join(img_dir, '*.tif'))
    # for path in img_list:
    #     mask = cv2.imread(path, -1)
    #     skl = skeletonize(mask / 255) * 255
    #
    #     cv2.imwrite(path.replace('masks', 'skl'), skl.astype(np.uint8))
    #
    #     print(path)


    img_dir = '/data/ldap_shared/home/s_lyr/code/noisyLabel/datasets/mito/train/images'
    mask_dir = '/data/ldap_shared/home/s_lyr/code/noisyLabel/datasets/mito/train/masks_randomFlip/ratio_20/v1'

    save_img_dir = '/data/ldap_shared/home/s_lyr/code/noisyLabel/datasets/mito/train/miniDataset/masks_randomFlip_ratio20/images_80'
    save_mask_dir = '/data/ldap_shared/home/s_lyr/code/noisyLabel/datasets/mito/train/miniDataset/masks_randomFlip_ratio20/masks_80'
    if not os.path.exists(save_img_dir): os.makedirs(save_img_dir)
    if not os.path.exists(save_mask_dir): os.makedirs(save_mask_dir)

    prob = 0.8

    path_list = glob.glob(os.path.join(mask_dir, "*.tif"))
    i = 0
    for path in path_list:

        img_name = os.path.split(path)[-1]
        print(img_name)

        x = np.random.rand()
        save_img_path = os.path.join(save_img_dir, img_name)
        save_mask_path = os.path.join(save_mask_dir, img_name)
        if x < prob:
            shutil.copy(path, save_mask_path)
            shutil.copy(os.path.join(img_dir, img_name), save_img_path)

            i += 1

    print(i)






