from torch import nn
from darknet53 import Darknet53

class PretrainHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin = nn.Linear(1024, 1000)

    def forward(self, x):
        avg = nn.functional.adaptive_max_pool2d(x.unsqueeze(0), output_size=1)
        activation = nn.functional.softmax(self.lin(avg))
        return activation
    

class ImageNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = Darknet53()
        self.head = PretrainHead()

    def forward(self, x):
        return self.head(self.backbone(x))
    