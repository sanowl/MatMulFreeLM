import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from bitlinear import BitLinear
from matmul_free_gru import MatMulFreeGRU

def train(model, data_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

if __name__ == "__main__":
    # Example Data
    inputs = torch.randn(100, 512)
    targets = torch.randn(100, 512)
    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Initialize model, criterion, optimizer
    model = BitLinear(in_features=512, out_features=512)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    train(model, data_loader, criterion, optimizer, num_epochs)
