import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

currentdir = os.path.dirname(__file__)
sys.path.append(currentdir)


from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone, output_stride=16, num_classes=1,
                 sync_bn=True, freeze_bn=False, is_res_loss=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.freeze_bn = freeze_bn

        self.dual_out = is_res_loss
        if self.dual_out:
            self.dual_decoder = build_decoder(num_classes, backbone, BatchNorm)

    def forward(self, input):
        x_0, low_level_feat = self.backbone(input)
        x_1 = self.aspp(x_0)
        x_2 = self.decoder(x_1, low_level_feat)
        x_3 = F.interpolate(x_2, size=input.size()[2:], mode='bilinear', align_corners=True)
        output = torch.sigmoid(x_3)

        if self.dual_out:
            d_x_0 = self.dual_decoder(x_1, low_level_feat)
            d_x_1 = F.interpolate(d_x_0, size=input.size()[2:], mode='bilinear', align_corners=True)
            daul_output = torch.sigmoid(d_x_1)

            return output, daul_output
        else:
            return output

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

class Discriminator(nn.Module):
    def __init__(self, input_channel, base_d = 64):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_channel, base_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_d, base_d * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_d*2, base_d*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_d*4, base_d*8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_d*8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d((1))
        )

    def forward(self, x):
        output = self.model(x)
        output = output.squeeze()

        return output


