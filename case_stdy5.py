import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]], dtype=torch.float32)

y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4)  # 2 inputs -> 4 hidden neurons
        self.output = nn.Linear(4, 1)  # 4 hidden -> 1 output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

model = XORNet()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


with torch.no_grad():
    predictions = model(X)
    predicted_labels = (predictions > 0.5).float()
    print("\nPredictions:")
    print(predicted_labels)
