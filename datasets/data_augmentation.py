# -*- coding: utf-8 -*-

import cv2
import numpy as np
from random import randint, random


def random_flip(image, mask):
    flip_seed = randint(-1,2)
    if flip_seed != 2:
        image = cv2.flip(image, flip_seed)
        mask = cv2.flip(mask, flip_seed)
    return image, mask


def random_rotation_scale(image, mask, angle_range):
    # scale = randint(90, 120) / 100 # scale from 0.7-1.3
    scale = 1
    angle = randint(angle_range[0], angle_range[1]) # rotation angle from [-45, 45]
    height, width = image.shape
    M = cv2.getRotationMatrix2D((height/2,width/2), angle, scale)
    out_image = cv2.warpAffine(image, M, (height,width))
    out_mask = cv2.warpAffine(mask, M, (height,width))
    out_image = clip_image(out_image)
    out_mask = convert_mask(out_mask)
    return out_image, out_mask


def random_shift(image, mask):
    shift_range = 30
    trans_x = randint(-1,1) * random() * shift_range # shift direction, shift percentage based on shift range
    trans_y = randint(-1,1) * random() * shift_range
    M = np.array([[1,0,trans_x],[0,1,trans_y]],dtype=np.float32)
    out_image = cv2.warpAffine(image, M, image.shape)
    out_mask = cv2.warpAffine(mask, M, mask.shape)
    return out_image, out_mask


def random_contrast(image):
    factor = randint(7,10) / 10
    mean = np.uint16(np.mean(image) + 0.5)
    mean_img = (np.ones(image.shape) * mean).astype(np.uint16)
    out_image = image.astype(np.uint16) * factor + mean_img * (1.0 - factor) 
    if factor < 0 or factor > 1:
        out_image = clip_image(out_image.astype(np.float))
    return out_image.astype(np.uint16)


def random_brightness(image):
    noise_scale = randint(7,13) / 10.
    noise_img = image * noise_scale
    out_image = clip_image(noise_img)
    return out_image


def random_noise(image):
    noise_seed = randint(0,1)
    if noise_seed == 0:
        noise_img = cv2.GaussianBlur(image, (5,5), 0)
    else:
        noise_img = image
    return noise_img


def clip_image(image):
    image[image > 65535.] = 65535
    image[image < 0.] = 0
    image = image.astype(np.uint16)
    return image


def convert_mask(mask):
    mask[mask >= 127.5] = 255
    mask[mask < 127.5] = 0 
    mask = mask.astype(np.uint8)
    return mask


def horizontal_flip(image, mask):
    image = cv2.flip(image, 1)
    mask = cv2.flip(mask, 1)
    return image, mask


def vertical_flip(image, mask):
    image = cv2.flip(image, 0)
    mask = cv2.flip(mask, 0)
    return image, mask


def rotation(image, mask, angle):
    height, width = image.shape
    M = cv2.getRotationMatrix2D((height/2,width/2), angle)
    out_image = cv2.warpAffine(image, M, (height, width))
    out_mask = cv2.warpAffine(mask, M, (height, width))
    return out_image, out_mask
