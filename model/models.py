import torch
import torch.nn as nn
import antialiased_cnns


class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1),
            antialiased_cnns.BlurPool(in_channels, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class UpLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpLayer, self).__init__()
        # Conv transpose upsampling

        self.blur_upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            antialiased_cnns.BlurPool(out_channels, stride=1)
        )

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.blur_upsample(x)
        x = torch.cat([x, skip], dim=1)  # Concatenate with skip connection
        return self.layer(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),  # output: 512 x 512 x 64
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # output: 512 x 512 x 64
            nn.LeakyReLU(inplace=True)
        )

        self.down1 = DownLayer(64, 128)  # output: 256 x 256 x 128
        self.down2 = DownLayer(128, 256)  # output: 128 x 128 x 256
        self.down3 = DownLayer(256, 512)  # output: 64 x 64 x 512
        self.down4 = DownLayer(512, 1024)  # output: 32 x 32 x 1024
        self.up1 = UpLayer(1024, 512)  # output: 64 x 64 x 512
        self.up2 = UpLayer(512, 256)  # output: 128 x 128 x 256
        self.up3 = UpLayer(256, 128)  # output: 256 x 256 x 128
        self.up4 = UpLayer(128, 64)  # output: 512 x 512 x 64
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)  # output: 512 x 512 x 3

    def forward(self, x):
        x0 = self.init_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.final_conv(x)
        return x


class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)
            # Output layer with 1 channel for binary classification
        )

    def forward(self, x):
        return self.model(x)
