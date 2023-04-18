from torch import nn
from torch.nn import functional

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 5)

    def __str__(self):
        return "simple_net"

    def forward(self, x):
        x = x.view(-1, 12288)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = functional.softmax(self.fc3(x), dim=-1)

        return x

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()

        self.fc1 = nn.Linear(12288, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 64)
        self.fc4 = nn.Linear(64, 5)

    def __str__(self):
        return "dense_net"

    def forward(self, x):
        x = x.view(-1, 12288)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = functional.relu(self.fc3(x))
        x = functional.softmax(self.fc4(x), dim=-1)

        return x

class SpectrogramRecognizer(nn.Module):
    def __init__(self):
        pass

    def __str__(self):
        return "spectr_recognizer"

    def forward(self, x):
        pass
