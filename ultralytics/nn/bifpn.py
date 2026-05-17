import numpy as np
import torch
import torch.nn as nn

class BiFPN(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  #normalize the weight
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)
    
class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon) 
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon) 
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)
    
class BiFPN_Add2(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add2, self).__init__()

        # Set learnable parameters
        # The role of nn.Parameter is to convert a non-trainable Tensor into a trainable Parameter,
        # and it registers this parameter inside the model.
        # That means model.parameters() will include this variable,
        # so it can be optimized automatically during training.
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        # Small value to avoid division by zero
        self.epsilon = 0.0001

        # 1×1 convolution to fuse the feature maps (change channel count from c1 → c2)
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)

        # SiLU activation (same as Swish: x * sigmoid(x))
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w  # learnable weights for the two feature inputs
        # Normalize weights so that their sum = 1 (soft weighting)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)

        # Weighted sum of the two inputs, followed by activation and 1×1 convolution
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1]))