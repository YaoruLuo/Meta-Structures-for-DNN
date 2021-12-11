from utils.optimize import configure_optimizers, soft_iou_loss, DMI_loss, DMI_IOU_loss
from iGTTmodule.generate_noise_label import randomShift_pixel, randomSample
from skimage.morphology import skeletonize
import torch


def emsModule(mask, shift_r, sample_p):
    '''
    :param mask: binary image
    :param shift_r: radius of shifting
    :param sample_p: probability of sampling
    :return: 8 bit image
    '''
    mask = skeletonize(mask) * 255
    mask = randomShift_pixel(mask, shift_r)
    mask = randomSample(mask, sample_p)
    return mask

def estimate_mask(preds, num_window):
    '''
    :param preds: tensor [b,c,h,w]
    :param num_window: number of spits in [thresh_min, thresh_max]
    :return: thresh_tensor [b,num_window, h, w]
            est_mask [b, num_window, h, w]
    '''
    b,c,h,w = preds.size()
    preds_flat = preds.view(b,c,-1)
    thresh_max, max_indice = torch.max(preds_flat, dim=-1)
    thresh_min, min_indice = torch.min(preds_flat, dim=-1)
    thresh_max = thresh_max.squeeze(1)
    thresh_min = thresh_min.squeeze(1)

    thresh_tensor = torch.zeros(b, num_window, h, w).cuda()
    for i in range(b):
        thresh_list = torch.linspace(thresh_min[i], thresh_max[i], num_window).unsqueeze(dim=-1).unsqueeze(dim=-1)
        thresh_tensor[i] = thresh_list.repeat(1,h,w)

    est_masks = torch.zeros_like(thresh_tensor).cuda()
    for j in range(num_window):
        thresh = thresh_tensor[:,j,:,:].unsqueeze(dim=1)
        est_masks[:,j:j+1,:,:] = torch.gt(preds, thresh).float()

    return thresh_tensor, est_masks


def find_best_estMask(preds, est_masks, criterion, ensemble=False):
    '''
    :param preds: predictions [b,1,h,w]
    :param est_masks: estimation masks [b,num_window,h,w]
    :param criterion: loss function of similarity calculation between pred and masks
    :return: best masks index, best masks [b,1,h,w]
    '''
    b,num_windiow,h,w = est_masks.size()
    preds_concat = preds.repeat(1,num_windiow,1,1)

    est_masks_flatten = est_masks.view(b, num_windiow, -1)
    preds_concat_flatten = preds_concat.view(b,num_windiow, -1)

    loss = 0
    best_estMask = torch.zeros_like(preds)

    if criterion == "dmi":
        dmi_loss = DMI_loss()
        loss = torch.zeros(num_windiow, b).cuda()
        for j in range(num_windiow)[1:-1]:
            loss[j] = dmi_loss(preds_concat[:,j:j+1,:,:], est_masks[:,j:j+1,:,:])
        loss = loss.transpose(0,1) # [b,num_window]

    elif criterion == "bce":
        loss = - (est_masks_flatten * torch.log(preds_concat_flatten) + (1 - est_masks_flatten) * torch.log(1 - preds_concat_flatten))
        loss = torch.sum(loss, dim=-1) # [b, num_window]

    elif criterion == "iou":
        iou_loss = soft_iou_loss()
        loss = torch.zeros(num_windiow, b).cuda()
        for j in range(num_windiow)[1:-1]:
            loss[j] = iou_loss(preds_concat[:, j:j+1, :, :], est_masks[:, j:j+1, :, :])
        loss = loss.transpose(0, 1)  # [b,num_window]

    min_loss_index = torch.argmin(loss, dim=-1)

    if ensemble:
        weight = torch.softmax(-1*loss, dim=-1)
        weight = weight.unsqueeze(dim=-1).unsqueeze(dim=-1)
        weight = weight.repeat(1,1,h,w)
        best_estMask = torch.mul(est_masks, weight)
        best_estMask = torch.sum(best_estMask, dim=1, keepdim=True)

    else:
        for i in range(min_loss_index.size()[-1]):
            best_estMask[i] = torch.index_select(est_masks[i], 0, min_loss_index[i])

    return min_loss_index, best_estMask
