import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.models import resnet50
from torchinfo import summary


def get_model(backbone):
    class FeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            # Change first conv layer to accept single-channel (grayscale) input
            # self.backbone.conv1.weight = torch.nn.Parameter(self.backbone.conv1.weight.sum(dim=1).unsqueeze(1))

        def forward(self, x):
            skip_connections = []
            for i in range(8):
                x = list(self.backbone.children())[i](x)
                if i in [2, 4, 5, 6, 7]:
                    skip_connections.append(x)
            encoder_outputs = skip_connections.pop(-1)
            skip_connections = skip_connections[::-1]

            return encoder_outputs, skip_connections

    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.dconv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.dconv(x)

    class UnetUpSample(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.convt = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2)
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.act1 = nn.ReLU(inplace=True)
            self.dconv = DoubleConv(out_channels * 2, out_channels)

        def forward(self, layer_input, skip_input):
            u = self.convt(layer_input)
            u = self.norm1(u)
            u = self.act1(u)
            u = torch.cat((u, skip_input), dim=1)
            u = self.dconv(u)
            return u

    class GateSignal(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.gate_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.gate_conv(x)

    class AttentionBlock(nn.Module):
        def __init__(self, gate_channels, x_channels, inter_channels):
            super().__init__()
            self.theta = nn.Conv2d(x_channels, inter_channels, kernel_size=2, stride=2)
            self.phi = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1)
            self.act1 = nn.ReLU(inplace=True)
            self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1)
            self.act2 = nn.Sigmoid()
            self.upsample = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
            self.final_conv = nn.Conv2d(inter_channels, x_channels, kernel_size=1, stride=1, bias=False)
            self.norm = nn.BatchNorm2d(x_channels)

        def forward(self, x, g):
            theta_x = self.theta(x)
            phi_g = self.phi(g)
            xg = torch.add(theta_x, phi_g)
            xg = self.act1(xg)
            score = self.psi(xg)
            score = self.act2(score)
            score = self.upsample(score)
            score = score.expand(x.shape)
            att_x = torch.mul(score, x)
            att_x = self.final_conv(att_x)
            att_x = self.norm(att_x)
            return att_x

    class UW2022AttentionUnet(nn.Module):
        def __init__(self, out_channels):
            super().__init__()
            self.encoder = FeatureExtractor()

            self.upsample1 = UnetUpSample(512, 256)
            self.upsample2 = UnetUpSample(256, 128)
            self.upsample3 = UnetUpSample(128, 64)
            self.upsample4 = UnetUpSample(64, 64)

            self.gate_signal1 = GateSignal(512, 256)
            self.gate_signal2 = GateSignal(256, 128)
            self.gate_signal3 = GateSignal(128, 64)
            self.gate_signal4 = GateSignal(64, 64)

            self.attention1 = AttentionBlock(256, 256, 256)
            self.attention2 = AttentionBlock(128, 128, 128)
            self.attention3 = AttentionBlock(64, 64, 64)
            self.attention4 = AttentionBlock(64, 64, 64)

            self.final_convt = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
            self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        def forward(self, x):
            x1, skip_connections = self.encoder(x)

            g1 = self.gate_signal1(x1)
            attention_skip1 = self.attention1(skip_connections[0], g1)
            x2 = self.upsample1(x1, attention_skip1)

            g2 = self.gate_signal2(x2)
            attention_skip2 = self.attention2(skip_connections[1], g2)
            x3 = self.upsample2(x2, attention_skip2)

            g3 = self.gate_signal3(x3)
            attention_skip3 = self.attention3(skip_connections[2], g3)
            x4 = self.upsample3(x3, attention_skip3)

            g4 = self.gate_signal4(x4)
            attention_skip4 = self.attention4(skip_connections[3], g4)
            x5 = self.upsample4(x4, attention_skip4)

            x6 = self.final_convt(x5)
            return self.final_conv(x6)

    model = UW2022AttentionUnet(out_channels=3)

    return model

# backbone = resnet50(pretrained=True)
# model = get_model(backbone)
#
# summary(
#     model,
#     input_size=(64, 3, 320, 384),
#     col_names=["output_size", "num_params"],
# )