import torch
import torch.nn as nn


class hw01_model(nn.Module):
    def __init__(self, inputDim, outputDim=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(inputDim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, outputDim),
        ).to("cuda:0")

    def forward(self, input):
        output = self.model(input)
        output = output.squeeze(1)
        return output

# model = hw01_model(inputDim=5,outputDim=1)
# input = [1,0.002,3,4,5]
# print(model(input))
