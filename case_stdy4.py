import torch
import torch.nn as nn
from torch.utils.data import DataLoader


ratings = torch.FloatTensor([
    [5, 3, -1],
    [4, -1, 2],
    [-1, 5, 3]
])

ratings[ratings >= 3] = 1
ratings[(ratings > 1) & (ratings < 3)] = 0

class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_hidden, num_visible)*0.1)
        self.h_bias = nn.Parameter(torch.zeros(num_hidden))
        self.v_bias = nn.Parameter(torch.zeros(num_visible))

    def sample_h(self, v):
        p = torch.sigmoid(v @ self.W.t() + self.h_bias)
        return p, torch.bernoulli(p)

    def sample_v(self, h):
        p = torch.sigmoid(h @ self.W + self.v_bias)
        return p, torch.bernoulli(p)

    def forward(self, v):
        _, h = self.sample_h(v)
        p_v, _ = self.sample_v(h)
        return p_v

num_visible = ratings.size(1)
num_hidden = 5
rbm = RBM(num_visible, num_hidden)
loader = DataLoader(ratings, batch_size=2, shuffle=True)

for epoch in range(10):
    for v0 in loader:
        ph0, h0 = rbm.sample_h(v0)
        v1_prob, v1 = rbm.sample_v(h0)
        ph1, _ = rbm.sample_h(v1)
        rbm.W.data += 0.01 * ((ph0.t() @ v0 - ph1.t() @ v1)/v0.size(0))
        rbm.v_bias.data += 0.01 * torch.sum(v0 - v1, dim=0)/v0.size(0)
        rbm.h_bias.data += 0.01 * torch.sum(ph0 - ph1, dim=0)/v0.size(0)


v = ratings[0].unsqueeze(0)
pred = rbm.forward(v)

mask = (v != -1)
pred[mask] = 0

top_movies = torch.topk(pred, 2).indices + 1
print("Top recommended movie IDs:", top_movies.flatten().numpy())

