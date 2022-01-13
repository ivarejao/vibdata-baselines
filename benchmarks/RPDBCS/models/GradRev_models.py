from torch import nn
from GradReversal import GradientReversal


class DomainAdapNetConv(nn.Module):
    def __init__(self, n_domains, n_classes, gradient_rev_lambda=1.0,
                 input_size=6100, output_size=8, activation_function=nn.ReLU()):
        super().__init__()
        self.convnet = nn.Sequential(  # input is (n_batches, 1, 6100)
            nn.Conv1d(1, 16, 5, padding=2), activation_function,  # 6100 -> 6096
            nn.Dropout(0.2),
            nn.MaxPool1d(4, stride=4),  # 6096 -> 1524
            nn.Conv1d(16, 32, 5, padding=2), activation_function,  # 1524 -> 1520
            nn.Dropout(0.2),
            nn.MaxPool1d(4, stride=4),  # 1520 -> 380
            nn.Conv1d(32, 64, 5, padding=2), activation_function,  # 380 -> 376
            nn.Dropout(0.2),
            nn.MaxPool1d(4, stride=4),  # 376 -> 94
            nn.Flatten(),
        )

        n = input_size//64
        self.encoder = nn.Sequential(nn.Linear(64 * n, 192),
                                     activation_function,
                                     nn.Linear(192, output_size)
                                     )

        self.domain_recognizer = nn.Sequential(
            GradientReversal(lambda_=gradient_rev_lambda),
            activation_function,
            nn.Linear(output_size, n_domains)
        )
        self.classifier = nn.Sequential(
            activation_function,
            nn.Linear(output_size, n_classes)
        )

    def forward(self, X, **kwargs):
        if(len(X.shape) == 2):
            X = X.reshape(len(X), 1, -1)
        Xe = self.encoder(self.convnet(X))
        Yd = self.domain_recognizer(Xe)
        Yc = self.classifier(Xe)

        return (Yd, Yc)


class DomainAdapNet(nn.Module):
    def __init__(self, n_domains, n_classes, gradient_rev_lambda=1.0,
                 input_size=6100, output_size=8, activation_function=nn.PReLU()):
        super(DomainAdapNet, self).__init__()
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

        self.domain_recognizer = nn.Sequential(
            GradientReversal(lambda_=gradient_rev_lambda),
            activation_function,
            nn.Linear(output_size, n_domains)
        )
        self.classifier = nn.Sequential(
            activation_function,
            nn.Linear(output_size, n_classes)
        )

    def forward(self, X, **kwargs):
        Xe = self.fc2(self.fc1(X))
        Yd = self.domain_recognizer(Xe)
        Yc = self.classifier(Xe)

        return (Yd, Yc)
