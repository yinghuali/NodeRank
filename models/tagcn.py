import torch
import torch.nn.functional as F
from torch_geometric.nn import TAGConv


class TAGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dic):
        super(TAGCN, self).__init__()
        self.conv1 = TAGConv(num_features, hidden_channels, **dic)
        self.conv2 = TAGConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class Target_TAGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = TAGConv(num_features, hidden_channels)
        self.conv2 = TAGConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

