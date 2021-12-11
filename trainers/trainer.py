import os
import sys
import time
import numpy as np
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
from visdom import Visdom

import models.unet as unet
import models.deeplabv3.deeplab_v3 as deeplab
import models.hrnet as hrnet

from datasets.data_loader import seg_dataloader
from utils.optimize import configure_optimizers, soft_iou_loss, DMI_loss, DMI_IOU_loss
from utils.utils import init_weights
from iGTTmodule.EMS import estimate_mask, find_best_estMask
from iGTTmodule.generate_noise_label import *

viz = Visdom(env="test", port=4004)

print("PyTorch Version: ", torch.__version__)

model_dict_segmentation = ["unet"]

def dice_coefficients(inputs, target):
    N = inputs.size()[0]
    dices = torch.zeros(N)
    for i in range(N):
        intersection = torch.sum(inputs[i, ...] * target[i, ...])
        input_area = torch.sum(inputs[i, ...])
        target_area = torch.sum(target[i, ...])
        dice = (2 * intersection) / (input_area + target_area + 1e-10)
        dices[i] = dice
    return dices

def find_threshold(scores, groundtruths):
    scores_max = torch.max(scores).cpu()
    scores_min = torch.min(scores).cpu()
    thresholds = np.linspace(scores_min, scores_max, 50)
    dices = np.zeros(len(thresholds))
    for indy in range(len(thresholds)):
        threshold = thresholds[indy]
        predictions = (scores >= threshold).float()
        dices[indy] = torch.mean(dice_coefficients(predictions, groundtruths))
    thred_indx = np.argmax(dices)
    return thresholds[thred_indx], dices[thred_indx], thresholds, dices


def evaluate(prediction, groundtruths):
    best_thred, mean_dice, thresh_list, dice_list = find_threshold(prediction, groundtruths)
    return best_thred, mean_dice, thresh_list, dice_list


class trainer_segmentation(nn.Module):
    def __init__(self, params=None):
        super(trainer_segmentation, self).__init__()
        self.args = params
        self.global_step = 0
        self.current_step = 0

    def _dataloader(self, datalist, shift_r, sample_p, is_iGTT, online_random = False, iGTT_replaceName=None, split='train'):
        dataset = seg_dataloader(img_list=datalist, split=split, is_normalization=self.args.normalization, shift_r = shift_r, sample_p = sample_p, is_iGTT=is_iGTT, online_random=online_random, replaceName=iGTT_replaceName)
        shuffle = True if split == 'train' else False
        if split == "train":
            batch_size = self.args.train_batch_size
        else:
            batch_size = self.args.val_batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=64, shuffle=shuffle,
                                drop_last=False)
        return dataloader

    def train_one_epoch(self, epoch):
        t0 = 0.0
        one_epoch_loss = 0.0

        self.model.train()

        # -----------------------------------
        for inputs in self.train_data_loader:

            self.global_step += 1
            self.current_step += 1

            t1 = time.time()

            images, masks, IDs = inputs["image"].cuda(), inputs["mask"].cuda(), inputs["ID"]

            if self.args.onlineRandom == True:
                f_flipRatio = 0.45
                # f_flipRatio = np.random.uniform(0, 0.5)
                b_flipRatio = 0.45
                # b_flipRatio = np.random.uniform(0, 0.5)
                print(f_flipRatio, b_flipRatio)

                masks[masks > 0] = 255
                masks = randomFlip(masks.squeeze(0).squeeze(0).cpu().numpy(), f_flipRatio=f_flipRatio, b_flipRatio=b_flipRatio)
                masks[masks > 0] = 1
                masks = torch.from_numpy(masks).unsqueeze(0).unsqueeze(0).float().cuda()

            prd = self.model(images)

            if self.args.is_iGTT:
                if self.args.segmentation_loss == "dmi" or self.args.segmentation_loss == "dmi_iou" or self.args.segmentation_loss == "dmi_bce":
                    loss = self.loss_func(prd, masks, is_cpu=True)
                else:
                    loss = self.loss_func(prd, masks)
            else:
                loss = self.loss_func(prd, masks)

            one_epoch_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            t0 += (time.time() - t1)

            if self.global_step % self.args.print_steps == 0:
                message = "Epoch: %d Step: %d LR: %.6f Total Loss: %.4f Runtime: %.2f s/%d iters." % (
                epoch + 1, self.global_step, self.lr_scheduler.get_lr()[-1], loss, t0, self.current_step)
                print("==> %s" % (message))
                with open(self.args.log_file, "a+") as fid:
                    fid.write('%s\n' % message)

                self.current_step = 0
                t0 = 0.0

                raw_img = images[0, 0, ...].data.cpu().numpy()
                mask = masks[0, 0, ...].data.cpu().numpy()
                pred_score = prd[0,0, ...].data.cpu().numpy()

                viz.heatmap(raw_img, win="train_image", opts={"title": "Train Image", "colormap": "Viridis"})
                viz.heatmap(pred_score, win="train prd score", opts={"title": "Train pred score", "colormap": "Viridis"})
                viz.image(mask, win="train_mask", opts={"title": "Train Mask"})

            if self.args.is_iGTT:
                num_window = self.args.num_window
                iGTT_save_path = self.args.iGTT_save_path

                thresh_tensor, est_masks = estimate_mask(prd.data, num_window)

                min_loss_index, best_estMasks = find_best_estMask(prd.data, est_masks, criterion=self.args.estMask_loss)

                for j in range(images.shape[0]):
                    image_name = IDs[j]

                    # save prediction images
                    pred = prd[j].data.cpu().numpy().squeeze()
                    prd_save_dir = os.path.join(iGTT_save_path, "numWindow_" + str(num_window), 'iter_' + str(epoch+1), 'pred')
                    if not os.path.exists(prd_save_dir): os.makedirs(prd_save_dir)
                    np.save(os.path.join(prd_save_dir, image_name).replace('tif', 'npy'), pred)

                    # save the best estimation mask
                    best_estMask = best_estMasks[j].data.cpu().squeeze().numpy() * 255

                    best_estMask_save_dir = os.path.join(iGTT_save_path, "numWindow_" + str(num_window), 'iter_' + str(epoch+1), 'bestMask')
                    if not os.path.exists(best_estMask_save_dir): os.makedirs(best_estMask_save_dir)
                    cv2.imwrite(os.path.join(best_estMask_save_dir, image_name), best_estMask.astype(np.uint8))


        mean_Eachepoch_loss = one_epoch_loss / len(self.train_data_loader)

        viz.line(Y=[mean_Eachepoch_loss], X=torch.Tensor([epoch + 1]), win='train loss', update='append',
                 opts=dict(title="Training Loss", xlabel="Epoch", ylabel="Train Loss"))

        # write message in text
        train_each_epoch_loss_message = "%.4f" % (mean_Eachepoch_loss)
        with open(os.path.join(self.args.text_message_dir, "train_val_epoch_loss.txt"), "a+") as fid:
            fid.write("%s\n" % train_each_epoch_loss_message)

        return mean_Eachepoch_loss


    def val_one_epoch(self, epoch):
        with torch.no_grad():
            self.model.eval()

            epoch_prd = 0.
            epoch_masks = 0.
            one_epoch_loss = 0.
            index = 0

            for i, inputs in enumerate(self.val_data_loader):
                images, masks = inputs["image"].cuda(), inputs["mask"].cuda()

                if self.args.predInvert:
                    prd = 1 - self.model(images)
                else:
                    prd = self.model(images)

                val_loss = self.loss_func(prd, masks)

                if i == 0:
                    epoch_prd = prd
                    epoch_masks = masks
                else:
                    epoch_prd = torch.cat((epoch_prd, prd), dim=0)
                    epoch_masks = torch.cat((epoch_masks, masks), dim=0)

                viz_img = images[0, 0, ...].data.cpu().numpy()
                viz_mask = masks[0,0, ...].data.cpu().numpy()
                viz_pred = prd[0,0, ...].data.cpu().numpy()

                # visdom visualize
                viz.heatmap(viz_img, win="val_image", opts={"title": "Val Image", "colormap": "Viridis"})
                viz.heatmap(viz_pred, win="val prd score",
                            opts={"title": "val pred score", "colormap": "Viridis"})
                viz.image(viz_mask, win="val_mask", opts={"title": "Val Mask"})


                one_epoch_loss += val_loss.item()
                index += 1

            best_thresh, mean_eachEpoch_dice, thresh_list, dice_list = evaluate(epoch_prd, epoch_masks)

            i_best_thresh, i_mean_eachEpoch_dice, i_thresh_list, i_dice_list = evaluate(1 - epoch_prd, epoch_masks)

            mean_eachEpoch_loss = one_epoch_loss / index


            viz.line(Y=[mean_eachEpoch_loss], X=torch.Tensor([epoch + 1]), win='val loss', update='append',
                     opts=dict(title="Validation Loss", xlabel="Epoch", ylabel="Val Loss"))

            viz.line(Y=[mean_eachEpoch_dice], X=torch.Tensor([epoch + 1]), win='val dice', update='append',
                     opts=dict(title="Validation Dice", xlabel="Epoch", ylabel="Val Dice"))

            viz.line(Y=[i_mean_eachEpoch_dice], X=torch.Tensor([epoch + 1]), win='invert val dice', update='append',
                     opts=dict(title="Invert Validation Dice", xlabel="Epoch", ylabel="Val Dice"))

            message = "Val Epoch: %d thresh: %.3f dice: %.4f invertPred dice: %.4f" % (
                epoch + 1, best_thresh, mean_eachEpoch_dice, i_mean_eachEpoch_dice)

            print("==> %s" % (message))

            # write message in text
            with open(self.args.log_file, "a+") as fid:
                fid.write('%s\n' % message)

            val_each_epoch_loss_message = "%.4f" % (mean_eachEpoch_loss)
            with open(os.path.join(self.args.text_message_dir, "train_val_epoch_loss.txt"), "a+") as fid:
                fid.write('%s\n' % val_each_epoch_loss_message)

            val_thresh_dice_message = "%.3f %.4f" % (best_thresh, mean_eachEpoch_dice)
            with open(os.path.join(self.args.text_message_dir, "val_best_threshold_dice.txt"), "a+") as fid:
                fid.write('%s\n' % val_thresh_dice_message)

            i_val_thresh_dice_message = "%.3f %.4f" % (best_thresh, i_mean_eachEpoch_dice)
            with open(os.path.join(self.args.text_message_dir, "val_best_threshold_invert_dice.txt"), "a+") as fid:
                fid.write('%s\n' % i_val_thresh_dice_message)

            with open(os.path.join(self.args.text_message_dir, "thresh_dice_list.txt"), "a+") as fid:
                for thresh in thresh_list:
                    fid.write("%.5f " % thresh)
                fid.write("\n")
                for dice in dice_list:
                    fid.write("%.4f " % dice)
                fid.write("\n")

        return mean_eachEpoch_loss, mean_eachEpoch_dice,i_mean_eachEpoch_dice, best_thresh, thresh_list, dice_list

    def train(self):
        print("==> Create model.")
        start_epoch = 0
        n_classes = self.args.num_class
        online_random = self.args.onlineRandom
        logits = True

        if self.args.network == "unet":
            self.model = unet.UNet(in_dim=1, n_classes=n_classes, logits=logits)

        elif self.args.network == "deeplabv3+":
            self.model = deeplab.DeepLab(backbone='resnet50', output_stride=16, num_classes=n_classes)

        elif self.args.network == "hrnet":
            self.model = hrnet.HRNetV2(self.args.input_channel, n_class=n_classes)

        else:
            print("No support model type.")
            sys.exit(1)

        init_weights(self.model)

        # load pretrained model
        if self.args.pretrain is not None and os.path.isfile(self.args.pretrain):
            print("==> Train from model '{}'".format(self.args.pretrain))
            checkpoint = torch.load(self.args.pretrain)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("==> Loaded checkpoint '{}')".format(self.args.pretrain))
            for param in self.model.parameters():
                param.requires_grad = True

        elif self.args.resume is not None and os.path.isfile(self.args.resume):
            print("==> Resume from checkpoint '{}'".format(self.args.resume))
            checkpoint = torch.load(self.args.resume)
            start_epoch = checkpoint['epoch'] + 1
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if
                               k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(pretrained_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("==> Loaded checkpoint '{}' (epoch {})".format(self.args.resume, checkpoint['epoch'] + 1))

        else:
            print("==> Train from initial or random state.")

        self.model.cuda()
        self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_list)

        print("==> List learnable parameters")
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                print("\t{}".format(name))

        print("==> Load data.")
        self.train_data_loader = self._dataloader(self.args.train_data_list, split='train', shift_r=None, sample_p=None, is_iGTT=False, online_random=online_random)
        self.val_data_loader = self._dataloader(self.args.val_data_list, split='val', shift_r=None, sample_p=None, is_iGTT=False, online_random=False)

        print("==> Configure optimizer.")
        self.optimizer, self.lr_scheduler = configure_optimizers(self.args.optimizer_type, self.model, self.args.init_lr, self.args.weight_decay,
                                                                 self.args.gamma, self.args.lr_decay_every_x_epochs)

        if self.args.segmentation_loss == 'ce':
            self.loss_func = nn.CrossEntropyLoss(reduction='mean')

        elif self.args.segmentation_loss == 'bce':
            self.loss_func = nn.BCELoss()

        elif self.args.segmentation_loss == "iou":
            self.loss_func = soft_iou_loss()

        elif self.args.segmentation_loss == "dmi":
            self.loss_func = DMI_loss()

        elif self.args.segmentation_loss == "dmi_iou":
            self.loss_func = DMI_IOU_loss()

        print("==> Start training")
        since = time.time()

        best_val_dice = 0
        i_best_val_dice = 0
        best_sample_epoch = 1

        for epoch in range(start_epoch, self.args.epochs):

            if self.args.is_iGTT and epoch > 0:
                num_window = self.args.num_window

                iGTT_replaceName = os.path.join(self.args.iGTT_dir, self.args.train_log_name,
                                                     "numWindow_" + str(num_window),
                                                    'iter_' + str(best_sample_epoch), 'bestMask')

                print("==> Best Mask from Epoch: ", best_sample_epoch)

                self.train_data_loader = self._dataloader(self.args.train_data_list, split='train', shift_r = self.args.shift_r, sample_p = self.args.sample_p, is_iGTT=True, online_random=False,
                                                          iGTT_replaceName=iGTT_replaceName)

            _ = self.train_one_epoch(epoch)
            val_mean_epoch_loss, val_mean_epoch_dice, i_val_mean_epoch_dice, best_thresh, thresh_list, dice_list = self.val_one_epoch(epoch)

            # use evaluation data to determine whether to update labels
            if self.args.is_iGTT and self.args.use_evaluation == True:
                if val_mean_epoch_dice > best_val_dice:
                    best_val_dice = val_mean_epoch_dice
                    best_sample_epoch = epoch + 1
                elif i_val_mean_epoch_dice > i_best_val_dice:
                    i_best_val_dice = i_val_mean_epoch_dice
                    best_sample_epoch = epoch + 1

            # without using evaluation data
            elif self.args.is_iGTT and self.args.use_evaluation == False:
                best_sample_epoch = epoch + 1

            if epoch >= self.args.start_save_ckpt and epoch % self.args.save_every_x_epochs == 0:
                torch.save({'epoch': epoch + 1,
                            'model_state_dict': self.model.module.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                           os.path.join(self.args.ckpt_dir, "epoch_" + str(epoch + 1) + ".pth"))
            self.lr_scheduler.step()

        print("==> Runtime: %.2f minutes." % ((time.time() - since) / 60.0))


if __name__ == "__main__":
    a = torch.FloatTensor([1, 0, 1])
