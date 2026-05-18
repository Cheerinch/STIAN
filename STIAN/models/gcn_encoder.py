import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):

    def __init__(
        self,
        in_dim=2,
        hidden_dim=64,
        out_dim=32
    ):

        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

        self.dropout = nn.Dropout(0.6)

    def forward(self, x, edge_index):

        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return x