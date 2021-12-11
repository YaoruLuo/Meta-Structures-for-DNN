import numpy as np
import glob
import cv2
import os
import math
import matplotlib.pyplot as plt

#-----------------------
# Basic function
#-----------------------

def randomSample(mask, sampleRatio):
    '''
    :param mask: ground truth of the image, size e.g. 256 x 256
    :param noise_ratio: sampling ratio of the mask, 0~1
    :return: noise label
    '''
    noisyLabel = np.zeros_like(mask)
    noneZeroCoordinate = np.transpose(np.nonzero(mask))
    sampleLength = len(noneZeroCoordinate)
    sampleList = np.arange(sampleLength)

    sample = np.random.choice(sampleList, math.ceil(sampleLength * sampleRatio), replace=False)

    for i in sample:
        sample_coordinate = noneZeroCoordinate[i]
        noisyLabel[sample_coordinate[0], sample_coordinate[1]] = 255

    return noisyLabel

def randomFlip(mask, f_flipRatio, b_flipRatio):
    forward = mask
    backward = 255 - mask

    foreground_noiseLabel = randomSample(forward, 1 - f_flipRatio)
    background_noiseLabel = randomSample(backward, 1 - b_flipRatio)

    noisyLabel = foreground_noiseLabel + (255 - background_noiseLabel) * (backward / 255)

    return noisyLabel


def randomGenerate(mask, ratio):
    '''
    :param mask: ground truth
    :param ratio: probability to set pixels as 1
    :return: random generate mask
    '''
    assert ratio >= 0 and ratio <= 1
    h,w = mask.shape

    if ratio == 1:
        mask_matrix = np.ones([h,w])
        mask_matrix = mask_matrix * 255
        return mask_matrix

    else:
        mask_matrix = np.random.uniform(0, 1, [h, w])
        mask_matrix[mask_matrix > ratio] = 255
        mask_matrix[mask_matrix <= ratio] = 0

        return 255 - mask_matrix


def randomShift_pixel(mask, shift_r):
    '''
    :param mask: [b,c,h,w]
    :param shift_r: shift radius [-r, r]
    :return: shifted mask
    '''
    h,w = mask.shape
    randomShift_img = np.zeros_like(mask)

    shift_list = np.arange(-shift_r, shift_r + 1, 1)

    noneZero_mask_coordinate = np.transpose(np.nonzero(mask))

    for coordinate in noneZero_mask_coordinate:
        shift_h = np.random.choice(shift_list, 1, replace=False)
        shift_w = np.random.choice(shift_list, 1, replace=False)

        new_h = coordinate[0] + shift_h
        if new_h >= h:
            new_h = h-1
        elif new_h < 0:
            new_h = 0

        new_w = coordinate[1] + shift_w
        if new_w >= w:
            new_w = w-1
        elif new_w < 0:
            new_w = 0

        randomShift_img[new_h, new_w] = 255

    return randomShift_img

#---------------------------
# Generate file function
#---------------------------

def generate_noise_label(mask_dir, save_dir, noise_type, f_ratioList = None, b_ratioList = None, generateRatio = None):
    '''
    :param mask_dir: mask_dir = './train/masks_aug_v1'
    :param save_dir: save_dir = './train/masks_aug_v1_randomSample'
    :param ratio_list: ratio_list = 0 ~ 100
    :param noise_type: "randomSample", "randomFlip", "randomGenerate"
    :return:
    '''
    mask_path_list = glob.glob(os.path.join(mask_dir, '*.tif'))
    for mask_path in mask_path_list:
        print(mask_path)
        img_name = os.path.split(mask_path)[-1]

        mask = cv2.imread(mask_path, -1)

        for j in range(1):
            if noise_type == 'randomSample':

                for i in range(len(f_ratioList)):
                    f_ratio = f_ratioList[i]
                    b_ratio = b_ratioList[i]

                    save_path = os.path.join(save_dir, "fRatio_" + str(f_ratio) + "_bRatio_" + str(b_ratio), "v" + str(j + 1))
                    if not os.path.exists(save_path): os.makedirs(save_path)

                    if f_ratio != 0:
                        noisyLabel = randomFlip(mask, (100 - f_ratio) * 0.01, b_ratio * 0.01)
                    else:
                        noisyLabel = randomFlip(mask, f_ratio * 0.01, (100 - b_ratio) * 0.01)

                    cv2.imwrite(os.path.join(save_path, img_name), noisyLabel.astype(np.uint8))

            elif noise_type == 'randomFlip':
                for i in range(len(f_ratioList)):
                    f_ratio = f_ratioList[i]
                    b_ratio = b_ratioList[i]

                    save_path = os.path.join(save_dir, "fRatio_" + str(f_ratio) + "_bRatio_" + str(b_ratio), "v" + str(j + 1))
                    if not os.path.exists(save_path): os.makedirs(save_path)

                    noisyLabel = randomFlip(mask, f_ratio * 0.01, b_ratio * 0.01)
                    cv2.imwrite(os.path.join(save_path, img_name), noisyLabel.astype(np.uint8))

            elif noise_type == 'randomGenerate':

                for ratio in generateRatio:

                    save_path = os.path.join(save_dir, "randomRatio_" + str(ratio),
                                             "v" + str(j + 1))
                    if not os.path.exists(save_path): os.makedirs(save_path)

                    noisyLabel = randomGenerate(mask, ratio * 0.01)
                    cv2.imwrite(os.path.join(save_path, img_name), noisyLabel.astype(np.uint8))

    return


if __name__ == '__main__':
    mask_dir = '...'
    save_dir = '...'
    foreground_ratio_list = [20]
    background_ratio_list = [20]
    generateRatio = [0,10,30,50]

    generate_noise_label(mask_dir, save_dir, noise_type='randomFlip', f_ratioList=foreground_ratio_list, b_ratioList=background_ratio_list)




