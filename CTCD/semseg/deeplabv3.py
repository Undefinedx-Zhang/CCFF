import torch
from torch import nn
import torch.nn.functional as F
from CTCD.semseg.base import BaseNet


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=True)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, bias=True)
        self.aspp5 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        size = x.size()[2:]
        x1 = self.relu(self.aspp1(x))
        x2 = self.relu(self.aspp2(x))
        x3 = self.relu(self.aspp3(x))
        x4 = self.relu(self.aspp4(x))

        x5 = self.aspp5(x)
        x5 = F.interpolate(self.fc(x5), size=size, mode='bilinear', align_corners=True)

        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out = self.dropout(out)
        return out


class DeepLabV3(BaseNet):
    def __init__(self, backbone, nclass):
        super(DeepLabV3, self).__init__(backbone)

        # Initialize ASPP
        self.aspp = ASPP(2048, 256)
        self.classifier = nn.Conv2d(256 * 5, nclass, kernel_size=1, stride=1, padding=0, bias=True)

        # Initialize weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)

    def base_forward(self, x):
        h, w = x.shape[-2:]

        # Extract feature from backbone
        x = self.backbone.base_forward(x)[-1]

        # Apply ASPP and classifier
        x = self.aspp(x)
        out = self.classifier(x)

        # Upsample to original size
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out
