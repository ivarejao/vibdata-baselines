from mixstyle import MixStyle
from torch import nn
import torch


class RPDBCS2020Net(nn.Module):
    """
    Network architecture used in https://doi.org/10.1109/IJCNN48605.2020.9207133
    """

    def __init__(self, input_size=6100, output_size=8, activation_function=nn.PReLU(), random_state=None):
        if(random_state is not None):
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)
        super().__init__()
        self.convnet = nn.Sequential(  # input is (n_batches, 1, 6100)
            nn.Conv1d(1, 16, 5, padding=2), activation_function,  # 6100 -> 6096
            nn.MaxPool1d(4, stride=4),  # 6096 -> 1524
            nn.Conv1d(16, 32, 5, padding=2), activation_function,  # 1524 -> 1520
            nn.MaxPool1d(4, stride=4),  # 1520 -> 380
            nn.Conv1d(32, 64, 5, padding=2), activation_function,  # 380 -> 376
            nn.MaxPool1d(4, stride=4),  # 376 -> 94
            nn.Flatten(),
        )

        n = input_size//4//4//4
        self.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(64 * n, 192),
                                activation_function,
                                nn.Linear(192, output_size)
                                )

        # print('>>>Number of parameters: ',sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        if(x.dim() == 2):
            x = x.reshape(x.shape[0], 1, x.shape[1])
        output1 = self.convnet(x)
        output2 = self.fc(output1)
        # NOTE: By default, skorch uses the first element of the tuple as the default output of net.predict() and net.predict_proba().
        return output2, output1


class RPDBCS2020NetWithMixStyle(nn.Module):
    """
    MixStyle: See https://arxiv.org/abs/2107.02053
    """

    def __init__(self, input_size=6100, output_size=8, activation_function=nn.PReLU(), random_state=None):
        if(random_state is not None):
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)
        super().__init__()
        self.mixstyle = MixStyle(p=0.5, alpha=0.1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), activation_function,  # 6100 -> 6096
            nn.MaxPool1d(4, stride=4)  # 6096 -> 1524
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, 5, padding=2), activation_function,  # 1524 -> 1520
            nn.MaxPool1d(4, stride=4),  # 1520 -> 380
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2), activation_function,  # 380 -> 376
            nn.MaxPool1d(4, stride=4),  # 376 -> 94
            nn.Flatten()
        )

        n = input_size//4//4//4
        self.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(64 * n, 192),
                                activation_function,
                                nn.Linear(192, output_size)
                                )

    def forward(self, x):
        if(x.dim() == 2):
            x = x.reshape(x.shape[0], 1, x.shape[1])
        x = self.conv1(x)
        x = self.mixstyle(x.reshape(-1, x.shape[1], x.shape[2], 1))
        x = self.conv2(x.reshape(-1, x.shape[1], x.shape[2]))
        x = self.mixstyle(x.reshape(-1, x.shape[1], x.shape[2], 1))
        x = self.conv3(x.reshape(-1, x.shape[1], x.shape[2]))
        x = self.fc(x)
        return x


class BigRPDBCS2020Net(nn.Module):
    def __init__(self, input_size=6100, output_size=8, activation_function=nn.PReLU(), random_state=None):
        if(random_state is not None):
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2), activation_function,
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(64, 128, 5, padding=2), activation_function,
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(128, 256, 5, padding=2), activation_function,
            nn.MaxPool1d(4, stride=4),
            nn.Flatten(),
        )

        n = input_size//4//4//4
        self.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(256 * n, 256),
                                activation_function,
                                nn.Linear(256, output_size)
                                )

        # print('>>>Number of parameters: ',sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        if(x.dim() == 2):
            x = x.reshape(x.shape[0], 1, x.shape[1])
        output = self.convnet(x)
        output = self.fc(output)
        return output


class FastRPDBCS2020Net(nn.Module):
    def __init__(self, input_size=6100, output_size=8, activation_function=nn.PReLU()):
        super().__init__()
        self.convnet = nn.Sequential(  # input is (n_batches, 1, 6100)
            nn.Conv1d(1, 16, 5, padding=2), activation_function,
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(16, 16, 5, padding=2), activation_function,
            nn.MaxPool1d(8, stride=8),
            nn.Flatten(),
        )

        n = input_size//64
        self.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(16 * n, 192), activation_function,
                                nn.Linear(192, output_size)
                                )

        # print('>>>Number of parameters: ',sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        if(x.dim() == 2):
            x = x.reshape(x.shape[0], 1, x.shape[1])
        output = self.convnet(x)
        output = self.fc(output)
        return output


class MLP6_backbone(nn.Module):
    """
    Adapted from https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark/blob/master/models/MLP.py
    """

    def __init__(self, input_size=6100, output_size=8, activation_function=nn.PReLU()):
        super(MLP6_backbone, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            activation_function,

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            activation_function,

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            activation_function,

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            activation_function,

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            activation_function,

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            activation_function)

        self.fc2 = nn.Linear(64, output_size)

    def forward(self, X):
        return self.fc2(self.fc1(X))


class SuperFast_backbone(nn.Module):
    """
    For debug purposes only.
    """

    def __init__(self, input_size=6100, output_size=8, activation_function=nn.PReLU()):
        super(SuperFast_backbone, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 2),
            activation_function)
        self.fc2 = nn.Linear(2, output_size)

    def forward(self, X):
        return self.fc2(self.fc1(X))


class CNN5(nn.Module):
    """
    Adapted from https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark/blob/master/models/CNN_1d.py
    """

    def __init__(self, input_size=6100, output_size=8, activation_function=nn.PReLU()):
        super(CNN5, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15),  # -14
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3),  # -2
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))  # /2

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),  # -2
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),  # -2
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4))

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True))
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        if(x.dim() == 2):
            x = x.reshape(x.shape[0], 1, x.shape[1])
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        x = self.fc(x)

        return x
