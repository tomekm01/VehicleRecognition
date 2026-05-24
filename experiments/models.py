import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck, ResNet50_Weights

# Attention modules

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=True),
            nn.Sigmoid()
        )
        nn.init.zeros_(self.fc[2].weight)
        nn.init.constant_(self.fc[2].bias, 1.0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CAM(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias=True),
        )
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.constant_(self.mlp[2].bias, 1.0)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = torch.nn.functional.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = torch.nn.functional.adaptive_max_pool2d(x, 1).view(b, c)
        channel_att = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return x * channel_att.view(b, c, 1, 1).expand_as(x)


class SAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=True)
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 1.0)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        concat = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = torch.sigmoid(self.conv(concat))
        return x * spatial_att.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.cam = CAM(channels=channels, r=r)
        self.sam = SAM()

    def forward(self, x):
        x = self.cam(x)
        x = self.sam(x)
        return x


# Custom bottlenecks

class SEBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__(inplanes, planes, stride, downsample, groups,
                         base_width, dilation, norm_layer)
        self.se = SEBlock(channels=planes * self.expansion, reduction=16)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CBAMBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__(inplanes, planes, stride, downsample, groups,
                         base_width, dilation, norm_layer)
        self.cbam = CBAM(channels=planes * self.expansion, r=16)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# Factory functions

def create_standard_resnet(num_classes):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    weights = ResNet50_Weights.IMAGENET1K_V2
    model.load_state_dict(weights.get_state_dict(), strict=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def create_se_resnet(num_classes):
    model = ResNet(SEBottleneck, [3, 4, 6, 3])
    weights = ResNet50_Weights.IMAGENET1K_V2
    model.load_state_dict(weights.get_state_dict(), strict=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def create_cbam_resnet(num_classes):
    model = ResNet(CBAMBottleneck, [3, 4, 6, 3])
    weights = ResNet50_Weights.IMAGENET1K_V2
    model.load_state_dict(weights.get_state_dict(), strict=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
