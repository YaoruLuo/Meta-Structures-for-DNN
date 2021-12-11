import sys
import torch
from torch.utils import data
import numpy as np
import cv2
import os

currentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(currentdir)

from iGTTmodule.EMS import emsModule
from iGTTmodule.generate_noise_label import *
from datasets.pre_processing import gray_norm


'''
Important for online refreshing masks !!!
BlackImg_dir_name: your dictionary name of black image masks
'''
BlackImg_dir_name = 'masks_Black_aug'


class seg_dataloader(data.Dataset):
    def __init__(self, img_list='', in_dim=1, split='train', in_size=256, out_size=256, is_normalization=False, shift_r=None, sample_p=None, is_iGTT=False, online_random=False, replaceName=None):
        super(seg_dataloader, self).__init__()
        self.split = split
        self.in_dim = in_dim
        self.in_size = in_size
        self.out_size = out_size
        self.is_norm = is_normalization
        self.shift_r = shift_r
        self.sample_p = sample_p
        self.is_iGTT = is_iGTT
        self.online_random = online_random
        self.replaceName = replaceName

        with open(img_list, "r") as fid:
            lines = fid.readlines()

        file_paths = []
        for line in lines:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(" ")
            if split == 'train' or split == 'val':
                file_paths.append((words[0], words[1]))
            elif split == 'test':
                file_paths.append(words[0])
        
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        sample = dict()
        if self.split == "train" or self.split == 'val':
            img_path, label_path = self.file_paths[index]

            if self.is_iGTT:

                label_path = label_path.replace(BlackImg_dir_name, self.replaceName)

            mask = cv2.imread(label_path, -1)

            # Can invert generated mask if foreground > background since background pixels account for a large proportion in most images
            # Just a trick to stabilize training, you can decide not to use it.
            if self.is_iGTT and self.split == 'train':
                mask = mask / 255
                if np.sum(mask) > (65535 / 2):
                    mask = 1 - mask

                # EMS
                mask = emsModule(mask, shift_r = self.shift_r, sample_p = self.sample_p)

            if self.out_size != self.in_size:
                mask = cv2.resize(mask, dsize=(self.out_size, self.out_size), interpolation=cv2.INTER_NEAREST)

            mask[mask>0] = 1

            sample["mask"] = torch.from_numpy(mask).unsqueeze(0).float()
        else:
            img_path = self.file_paths[index]
        
        # prepare images
        orig_img = cv2.imread(img_path, -1)
        img = self.img_preprocessing(orig_img, is_norm=self.is_norm)

        if self.in_size != self.out_size:
            img = cv2.resize(img, dsize=(self.out_size, self.out_size), interpolation=cv2.INTER_LINEAR)

        sample["image"] = torch.from_numpy(img).float().unsqueeze(0)
        sample["orig_img"] = torch.from_numpy(orig_img * 1.0).unsqueeze(0)
        sample["ID"] = os.path.split(img_path)[-1]

        return sample

    def img_preprocessing(self, img, is_norm):
        if img.dtype == "uint8":
            img = img / 255.
        elif img.dtype == "uint16":
            img = img / 65535.

        if is_norm:
            img = gray_norm(img)
        return img

        
if __name__ == "__main__":
    print('a')

