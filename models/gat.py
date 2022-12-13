import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.datasets import Planetoid


class GAT(torch.nn.Module):
    def __init__(self, features, hidden, classes, dic):
        super(GAT, self).__init__()
        self.gat1 = GATConv(features, hidden, **dic)
        self.gat2 = GATConv(hidden*dic['heads'], classes)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1)


class Target_GAT(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads=4):
        super(Target_GAT, self).__init__()
        self.gat1 = GATConv(features, hidden, heads=4)
        self.gat2 = GATConv(hidden*heads, classes)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1)
