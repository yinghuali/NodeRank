import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid
from config import *
from utils import *
from sklearn.model_selection import train_test_split

path_x_np = './data/lastfm/x_np.pkl'
path_edge_index = './data/lastfm/edge_index_np.pkl'
path_y = './data/lastfm/y_np.pkl'
epochs_list = epochs_tagcn
dic_mutation = dic_mutation_tagcn
path_save_model = 'mutation_models/lastfm_tagcn/lastfm_tagcn_'
# path_save_config = '/Users/yinghua.li/Documents/Pycharm/GNNEST/mutation/mutation_models/lastfm_tagcn/lastfm_tagcn_'
path_save_config = '/mnt/irisgpfs/users/yili/pycharm/GNNEST/mutation/mutation_models/lastfm_tagcn/lastfm_tagcn_'

model_name = 'tagcn'



def get_data():
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    return x, y, edge_index, num_node_features, num_classes, train_idx, test_idx


def train(hidden_channel, x, y, edge_index, num_node_features, num_classes, train_idx, test_idx, save_model_name, epochs, dic):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = select_model(model_name, hidden_channel, num_node_features, num_classes, dic)
    model = model.to(device)
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


def main(dic):
    x, y, edge_index, num_node_features, num_classes, train_idx, test_idx = get_data()
    j = 0
    for epochs in epochs_list:
        list_dic = list(ParameterGrid(dic))
        for i in hidden_channel_list:
            for tmp_dic in list_dic:
                save_model_name = path_save_model + str(i) + '_' + str(j) + '.pt'
                pickle.dump(tmp_dic, open(path_save_config + str(i) + '_' + str(j) + '.pkl', 'wb'))
                train(i, x, y, edge_index, num_node_features, num_classes, train_idx, test_idx, save_model_name, epochs, tmp_dic)
                j += 1


if __name__ == '__main__':
    main(dic_mutation)

