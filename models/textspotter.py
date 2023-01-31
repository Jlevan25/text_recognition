from torch import nn


class DeepTextSpotter(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()

        act = nn.LeakyReLU()

        class Block(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels=channels // 2, out_channels=channels, kernel_size=3, padding=1)
                self.norm = nn.BatchNorm2d(channels)
                self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

            def forward(self, x):
                x = act(self.conv1(x))
                x = self.norm(x)
                x = act(self.conv2(x))
                return x

        self.conv1_1 = nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        for i, channel in enumerate([64, 128, 256]):
            setattr(self, f'block{i+2}', Block(channel))

        self.conv5_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(2, 3), padding=(0, 1))
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 5), padding=(0, 2))

        self.conv_out = nn.Conv2d(in_channels=512, out_channels=out_channel, kernel_size=(1, 7), padding=(0, 3))

        self.max_pool12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool34 = nn.MaxPool2d(kernel_size=2, stride=(2, 1))
        self.pad = nn.ZeroPad2d((0, 1, 0, 0))

        self.act = act
        self.out_act = nn.LogSoftmax(1)

    def forward(self, input_):
        # 1st
        x = self.act(self.conv1_1(input_))
        x = self.act(self.conv1_2(x))
        x = self.max_pool12(x)

        # 2nd
        x = self.block2(x)
        x = self.max_pool12(x)

        # 3d
        x = self.block3(x)
        x = self.pad(x)
        x = self.max_pool34(x)

        # 4th
        x = self.block4(x)
        x = self.pad(x)
        x = self.max_pool34(x)

        # 5th
        x = self.act(self.conv5_1(x))
        x = self.act(self.conv5_2(x))

        logits = self.conv_out(x).squeeze(2)

        return self.out_act(logits), logits


if __name__ == '__main__':
    from torchsummary import summary

    model = DeepTextSpotter(1, 33)

    print(summary(model=model, input_size=(1, 32, 4)))
