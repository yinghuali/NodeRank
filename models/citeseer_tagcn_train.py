from tagcn import TAGCN
import pickle
import pickle
import numpy as np
import torch
import argparse
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--path_x_np", type=str)
ap.add_argument("--path_edge_index", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--epochs", type=int)
ap.add_argument("--save_model_name", type=str)
ap.add_argument("--save_pre_name", type=str)

args = ap.parse_args()
path_x_np = args.path_x_np
path_edge_index = args.path_edge_index
path_y = args.path_y
epochs = args.epochs
save_model_name = args.save_model_name
save_pre_name = args.save_pre_name

x = pickle.load(open(path_x_np, 'rb'))
edge_index = pickle.load(open(path_edge_index, 'rb'))
y = pickle.load(open(path_y, 'rb'))

num_node_features = len(x[0])
num_classes = len(set(y))
idx_np = np.array(list(range(len(x))))
train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.3, random_state=17)

x = torch.from_numpy(x)
edge_index = torch.from_numpy(edge_index)
y = torch.from_numpy(y)
train_y = torch.from_numpy(train_y)
test_y = torch.from_numpy(test_y)


def train(x, edge_index, y):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TAGCN(num_node_features, 16, num_classes).to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.nll_loss(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), save_model_name)

    model.eval()
    pred = model(x, edge_index).argmax(dim=1)

    correct = (pred[train_idx] == y[train_idx]).sum()
    acc = int(correct) / len(train_idx)
    print('train:', acc)

    correct = (pred[test_idx] == y[test_idx]).sum()
    acc = int(correct) / len(test_idx)
    print('test:', acc)

    pre = model(x, edge_index)
    pickle.dump(pre.detach().cpu().numpy(), open(save_pre_name, 'wb'), protocol=4)


if __name__ == '__main__':
    train(x, edge_index, y)



