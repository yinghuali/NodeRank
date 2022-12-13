import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input_dim, hiden_dim, output_dim):
        super(DNN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hiden_dim)
        self.linear2 = nn.Linear(hiden_dim, output_dim)

    def forward(self, x):
        hidden = self.linear1(x)
        activate = torch.relu(hidden)
        output = self.linear2(activate)
        output = torch.sigmoid(output)
        return output


def get_acc(outputs, labels):
    _, predict = torch.max(outputs.data, 1)
    total_num = labels.shape[0]*1.0
    correct_num = (labels == predict).sum().item()
    acc = correct_num / total_num

    return acc
