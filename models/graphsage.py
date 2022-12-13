import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.datasets import Planetoid


class GraphSAGE(torch.nn.Module):
    def __init__(self, feature, hidden, classes, dic):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(feature, hidden, **dic)
        self.sage2 = SAGEConv(hidden, classes)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)

        return F.log_softmax(x, dim=1)


class Target_GraphSAGE(torch.nn.Module):
    def __init__(self, feature, hidden, classes):
        super(Target_GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(feature, hidden)
        self.sage2 = SAGEConv(hidden, classes)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)

        return F.log_softmax(x, dim=1)



