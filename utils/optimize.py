   
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

   
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']  

def configure_optimizers(optimizer_type, model, lr, weight_decay, gamma, lr_decay_every_x_epochs):
    if optimizer_type == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_decay_every_x_epochs, gamma=gamma)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=lr_decay_every_x_epochs)
    return optimizer, scheduler


class soft_iou_loss(nn.Module):
    def __init__(self):
        super(soft_iou_loss, self).__init__()

    def forward(self, pred, label, k = 2, is_cpu=False):
        if is_cpu:
            pred = pred.cpu()
            label = label.cpu()

        b = pred.size()[0]
        pred = pred.view(b, -1)
        label = label.view(b, -1)
        inter = torch.sum(torch.mul(pred, label), dim=-1, keepdim=False)
        unit = torch.sum(torch.pow(pred, k) + label, dim=-1, keepdim=False) - inter
        return torch.mean(1 - inter / (unit + 1e-10))


class DMI_loss(nn.Module):
    def __init__(self):
        super(DMI_loss, self).__init__()

    def forward(self, input, target, is_cpu = False):

        if is_cpu:
            input = input.cpu()
            target = target.cpu()

        b,c,h,w = input.size()
        input_f = input.view(b,c,-1)
        input_b = 1 - input_f
        preds = torch.cat((input_f, input_b), dim=1)

        target_f = target.view(b,c,-1)
        target_b = 1 - target_f
        labels = torch.cat((target_f, target_b), dim=1)

        mat = torch.matmul(preds, labels.transpose(1,2))
        loss = - 1.0 * torch.log(torch.abs(torch.det(mat)) + 0.001)
        loss = torch.mean(loss)

        if is_cpu:
            loss = loss.cuda()

        return loss

class DMI_IOU_loss(nn.Module):
    def __init__(self):
        super(DMI_IOU_loss, self).__init__()

    def forward(self, input, target, is_cpu=False):
        dmiLoss = DMI_loss()
        iouLoss = soft_iou_loss()

        dmi_iou_loss = dmiLoss(input, target, is_cpu) + iouLoss(input, target)

        return dmi_iou_loss

class regress_loss(nn.Module):
    def __init__(self):
        super(regress_loss, self).__init__()

    def forward(self, x, y):
        b = x.size()[0]
        x = x.view(b, -1)
        y = y.view(b, -1)
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

if __name__ == "__main__":
    torch.manual_seed(seed=1)

    preds = torch.sigmoid(torch.randn(([2,1,256,256])))
    label = torch.ones((2,1,256,256))

    dmi_loss = soft_iou_loss()
    loss = dmi_loss(preds, label, k=10)
    print(loss)


