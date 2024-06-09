import torch
import torch.nn as nn

class MatMulFreeGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MatMulFreeGRU, self).__init__()
        self.hidden_size = hidden_size
        
        # Define the layers with ternary weights
        self.w_ir = nn.Parameter(torch.Tensor(hidden_size, input_size).sign())
        self.w_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size).sign())
        self.w_iz = nn.Parameter(torch.Tensor(hidden_size, input_size).sign())
        self.w_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size).sign())
        self.w_in = nn.Parameter(torch.Tensor(hidden_size, input_size).sign())
        self.w_hn = nn.Parameter(torch.Tensor(hidden_size, hidden_size).sign())
        
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            nn.init.kaiming_uniform_(weight, a=5**0.5)

    def forward(self, x, h):
        r = torch.sigmoid(torch.addmm(self.w_ir, x.t(), self.w_hr, h.t()).t())
        z = torch.sigmoid(torch.addmm(self.w_iz, x.t(), self.w_hz, h.t()).t())
        n = torch.tanh(torch.addmm(self.w_in, x.t(), r * torch.addmm(self.w_hn, h.t())).t())
        h = (1 - z) * n + z * h
        return h
