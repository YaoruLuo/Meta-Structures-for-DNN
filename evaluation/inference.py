import os
import sys
import numpy as np
import time
import argparse
import cv2

import torch
from torch.utils.data import DataLoader


currentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(currentdir)

import models.unet as unet
import models.deeplabv3.deeplab_v3 as deeplab
import models.hrnet as hrnet

from datasets.data_loader import seg_dataloader

def test(args):
    datatype = args.datatype
    batch_size = args.batch_size
    split = args.split
    test_ckpt_epoch = args.test_ckpt_epoch
    in_size = args.in_img_size
    out_size = args.out_img_size

    norm = args.normalization


    if args.datatype == 'er':
        data_txt = os.path.join(currentdir, 'dataset/txt/er', args.data_list)
    elif args.datatype == 'mito':
        data_txt = os.path.join(currentdir, 'dataset/txt/mito', args.data_list)

    os.environ['CUDA_VISIBLE_DEVICE'] = args.gpus
    gpu_list = []
    for str_id in args.gpus.split(','):
        id = int(str_id)
        gpu_list.append(id)
    args.gpu_list = gpu_list

    train_dir = args.train_dir


    ckpt_file = os.path.join(currentdir, "train_log", datatype, train_dir, "checkpoints", test_ckpt_epoch)
    if not os.path.exists(ckpt_file): ckpt_file = os.path.join(currentdir, "train_log", datatype, "iGTT", train_dir, "checkpoints", test_ckpt_epoch)

    print("==> Create model.")
    if args.network == "unet":
        model = unet.UNet(in_dim=1, n_classes=1)

    elif args.network == "deeplabv3+":
        model = deeplab.DeepLab(backbone='resnet50', output_stride=16, num_classes=1)

    elif args.network == "hrnet":
        model = hrnet.HRNetV2(in_channels=1, n_class=1)


    else:
        print("No support model type.")
        sys.exit(1)

    print("==> Load data.")
    if not os.path.exists(data_txt):
        print("No file list found.")
        sys.exit(1)
    with open(data_txt, 'r') as fid:
        lines = fid.readlines()

    test_dataset = seg_dataloader(img_list=data_txt, split=split, in_size=in_size,
                                     out_size=out_size, is_normalization=norm, shift_r = None, sample_p = None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)

    result_dir = ckpt_file.replace('checkpoints', 'results').split('.')[0]

    if not os.path.exists(result_dir): os.makedirs(result_dir)

    print("==> Load weights %s." % (ckpt_file))
    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=args.gpu_list)
    model.eval()

    t1 = 0
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):

            images, IDs = inputs["image"].cuda(), inputs["ID"]
            t0 = time.time()

            preds = model(images)

            t1 += (time.time() - t0)

            for j in range(images.shape[0]):
                image_name = IDs[j]
                pred = preds[j].data.cpu().numpy().squeeze()
                prd_save_dir = os.path.join(result_dir, 'pred')
                if not os.path.exists(prd_save_dir): os.makedirs(prd_save_dir)
                cv2.imwrite(os.path.join(prd_save_dir, image_name), pred)


        print('Runtime: %.2f ms/image' % (t1 * 1000 / len(lines)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--datatype', type=str, default='mito')

    parser.add_argument('--network', type=str, default='unet')
    parser.add_argument('--train_dir', type=str, default='unet_bce_sgd_norm_skl_aug_v1_20201123')
    parser.add_argument('--test_ckpt_epoch', type=str, default='checkpoints_epoch_30.pth')
    parser.add_argument('--data_list', type=str, default='test.txt')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--in_img_size', type=int, default=256)
    parser.add_argument('--out_img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpus', type=str, default="0,1,2,3")

    parser.add_argument("--normalization", action='store_true', default=True)

    args = parser.parse_args()

    test(args)
