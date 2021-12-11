import os, sys
import torch
import torch.nn as nn

currentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(currentdir)

from utils.base_module import DoubleConv, Down, Up, OutConv, SELayer, ConvLayer


class UNet(nn.Module):
    def __init__(self, in_dim=1, base_dim=64, n_classes=1, bilinear=False, logits=True, fuse_model = 'cat'):
        super(UNet, self).__init__()
        self.n_channels = in_dim
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.fuse_mode = fuse_model
        self.logits = logits
        self.feature_map = {}

        # encoder
        self.inc = DoubleConv(in_dim, base_dim)
        self.down1 = Down(base_dim, base_dim * 2)
        self.down2 = Down(base_dim * 2, base_dim * 4)
        self.down3 = Down(base_dim * 4, base_dim * 8)
        self.down4 = Down(base_dim * 8, base_dim * 16)

        # decoder
        self.up1 = Up(base_dim * 16, base_dim * 8, bilinear, fuse_mode=self.fuse_mode)
        self.up2 = Up(base_dim * 8, base_dim * 4, bilinear, fuse_mode=self.fuse_mode)
        self.up3 = Up(base_dim * 4, base_dim * 2, bilinear, fuse_mode=self.fuse_mode)
        self.up4 = Up(base_dim * 2, base_dim, bilinear, fuse_mode=self.fuse_mode)
        self.out = OutConv(base_dim, n_classes)


    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        o_4 = self.up1(x5, x4)
        o_3 = self.up2(o_4, x3)
        o_2 = self.up3(o_3, x2)
        o_1 = self.up4(o_2, x1)
        o_seg = self.out(o_1)

        if self.logits:
            if self.n_classes > 1:
                pred = torch.softmax(o_seg, dim=1)
            else:
                pred = torch.sigmoid(o_seg)
        else:
            pred = o_seg

        # self.feature_map['Down Layer 1'] = x1
        # self.feature_map['Down Layer 2'] = x2
        # self.feature_map['Down Layer 3'] = x3
        # self.feature_map['Down Layer 4'] = x4
        # self.feature_map['Down Layer 5'] = x5
        # self.feature_map['Up Layer 4'] = o_4
        # self.feature_map['Up Layer 3'] = o_3
        # self.feature_map['Up Layer 2'] = o_2
        # self.feature_map['Up Layer 1'] = o_1

        # for viz feature
        # return pred, self.feature_map
        return pred


class lightUNet(nn.Module):
    def __init__(self, in_dim=1, base_dim=64, n_classes=1, bilinear=False, logits=True, fuse_model = 'cat'):
        super(lightUNet, self).__init__()
        self.n_channels = in_dim
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.fuse_mode = fuse_model
        self.logits = logits

        # encoder
        self.inc = DoubleConv(in_dim, base_dim)
        self.down1 = Down(base_dim, base_dim * 2)
        self.down2 = Down(base_dim * 2, base_dim * 4)
        self.down3 = Down(base_dim * 4, base_dim * 8)
        # self.down4 = Down(base_dim * 8, base_dim * 16)

        # decoder
        # self.up1 = Up(base_dim * 16, base_dim * 8, bilinear, fuse_mode=self.fuse_mode)
        self.up2 = Up(base_dim * 8, base_dim * 4, bilinear, fuse_mode=self.fuse_mode)
        self.up3 = Up(base_dim * 4, base_dim * 2, bilinear, fuse_mode=self.fuse_mode)
        self.up4 = Up(base_dim * 2, base_dim, bilinear, fuse_mode=self.fuse_mode)
        self.out = OutConv(base_dim, n_classes)


    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)

        # o_4 = self.up1(x5, x4)
        o_3 = self.up2(x4, x3)
        o_2 = self.up3(o_3, x2)
        o_1 = self.up4(o_2, x1)
        o_seg = self.out(o_1)

        if self.logits:
            if self.n_classes > 1:
                pred = torch.softmax(o_seg, dim=1)
            else:
                pred = torch.sigmoid(o_seg)
        else:
            pred = o_seg

        return pred


if __name__ == "__main__":
    image = torch.randn([4,1,256,256])

    # model = lightUNet()
    model = UNet()
    output, feature_map = model(image)
    for key in feature_map.keys():
        feature = feature_map[key]
        print(key, feature)

    # print(output.size())
