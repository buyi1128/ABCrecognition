import torch
import math
import torch.nn as nn


class NewNet(nn.Module):
    def __init__(self):
        super(NewNet, self).__init__()
        self.conv1_1 = nn.Sequential(     # input_size=(1*28*28)
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32)
        )
        self.conv1_2 = nn.Sequential(  # input_size=(16*28*28)
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),  # padding=2保证输入输出尺寸相同
            nn.LeakyReLU(0.1),  # input_size=(6*24*24)
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)  # output_size=(6*14*14)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)  # output_size=()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 8 * 6, 128),
            # nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(128, 4)
        self.softmax = nn.Softmax(dim=-1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # 定义前向传播过程，输入为x
    def forward(self, x):
        # print(x.size())
        x = self.conv1_1(x)
        # print("conv1_1: ", x.size())
        x = self.conv1_2(x)
        # print("conv1_2:", x.size())
        x = self.conv2_1(x)
        # print("conv2_1:", x.size())
        x = self.conv2_2(x)
        # print("conv2_2: ", x.size())
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        # print("fc2: ", x.size())
        return x
        # return self.softmax(x)



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     # input_size=(1*36*16)
            nn.Conv2d(1, 6, 5, 1, 2),   # padding=2保证输入输出尺寸相同
            nn.ReLU(),      # input_size=(6*36*16)
            nn.MaxPool2d(kernel_size=2, stride=2)  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 4)
        self.softmax = nn.LogSoftmax(dim=1)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.size())
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        # print(x.size())
        x = self.fc1(x)
        # print(x)
        x = self.fc2(x)
        # print(x)
        x = self.fc3(x)
        # print(x)
        # print( self.softmax(x))
        return self.softmax(x)

# net = NewNet()
# x = torch.randn(4, 1, 36, 30)
# y = net(x)
# print(y)