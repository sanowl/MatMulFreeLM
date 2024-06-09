import torch
import torch.nn as nn

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, input):
        # Quantize weights to {-1, 0, +1}
        ternary_weight = self.weight.sign()
        return nn.functional.linear(input, ternary_weight)
