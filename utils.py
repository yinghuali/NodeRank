import os
import torch
import numpy as np
import pickle
import torch
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from models.gcn import GCN, Target_GCN
from models.gat import GAT, Target_GAT
from models.tagcn import TAGCN, Target_TAGCN
from models.graphsage import GraphSAGE, Target_GraphSAGE


def get_model_path(path_dir_compile):
    model_path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.pt'):
                    model_path_list.append(file_absolute_path)
    return model_path_list


def load_model(model_name, path_model, hidden_channel, num_node_features, num_classes, dic):
    if model_name == 'gcn':
        model = GCN(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'gat':
        model = GAT(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'graphsage':
        model = GraphSAGE(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'tagcn':
        model = TAGCN(num_node_features, hidden_channel, num_classes, dic)
    model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
    model.eval()
    return model


def load_target_model(model_name, num_node_features, hidden_channel, num_classes, target_model_path):
    if model_name == 'gcn':
        model = Target_GCN(num_node_features, hidden_channel, num_classes)
    elif model_name == 'gat':
        model = Target_GAT(num_node_features, hidden_channel, num_classes)
    elif model_name == 'graphsage':
        model = Target_GraphSAGE(num_node_features, hidden_channel, num_classes)
    elif model_name == 'tagcn':
        model = Target_TAGCN(num_node_features, hidden_channel, num_classes)
    model.load_state_dict(torch.load(target_model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def select_model(model_name, hidden_channel, num_node_features, num_classes, dic):
    if model_name == 'gcn':
        model = GCN(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'gat':
        model = GAT(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'graphsage':
        model = GraphSAGE(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'tagcn':
        model = TAGCN(num_node_features, hidden_channel, num_classes, dic)
    return model


def get_idx_miss_class(target_pre, test_y):
    idx_miss_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != test_y[i]:
            idx_miss_list.append(i)
    idx_miss_list.append(i)
    return idx_miss_list


def get_n_kill_model(target_pre, mutation_pre_list):
    n_kill_model = []
    for i in range(len(target_pre)):
        n = 0
        for j in range(len(mutation_pre_list)):
            if mutation_pre_list[j][i] != target_pre[i]:
                n += 1
        n_kill_model.append(n)
    return n_kill_model


def get_res_ratio_list(idx_miss_list, select_idx_list, select_ratio_list):
    res_ratio_list = []
    for i in select_ratio_list:
        n = round(len(select_idx_list) * i)
        tmp_select_idx_list = select_idx_list[: n]
        n_hit = len(np.intersect1d(idx_miss_list, tmp_select_idx_list, assume_unique=False, return_indices=False))
        ratio = round(n_hit / len(idx_miss_list), 4)
        res_ratio_list.append(ratio)
    return res_ratio_list


def load_data(path_x_np, path_edge_index, path_y):
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

    return num_node_features, num_classes, x, edge_index, y, test_y, train_y, train_idx, test_idx


def get_mutation_model_features(num_node_features, target_hidden_channel, num_classes, target_model_path, x, y, edge_index, model_list, model_name):
    target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, target_model_path)
    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()
    mutation_pre_idx_np = np.array([model(x, edge_index).argmax(dim=1).numpy() for model in model_list]).T
    feature_list = []
    for i in range(len(target_pre)):
        tmp_list = []
        for j in range(len(mutation_pre_idx_np[i])):
            if mutation_pre_idx_np[i][j] != target_pre[i]:
                tmp_list.append(1)
            else:
                tmp_list.append(0)
        feature_list.append(tmp_list)
    feature_np = np.array(feature_list)

    label_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != y[i]:
            label_list.append(1)
        else:
            label_list.append(0)
    label_np = np.array(label_list)

    return feature_np, label_np


def get_repeat_mutation_model_features(num_node_features, target_hidden_channel, num_classes, target_model_name, x, y, edge_index, repeat_path_mutation_model, model_name):
    """
    exampel:
    repeat_mutation_model_path: ./new_mutation_models/
    target_model_name: citeseer_gcn
    """
    repeat_path_target_model_list = ['./repeat_target_models/'+'repeat_'+str(i)+'/'+target_model_name+'.pt' for i in range(1, 21)] # 路径如下
    # ['./repeat_target_models/repeat_1/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_2/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_3/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_4/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_5/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_6/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_7/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_8/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_9/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_10/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_11/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_12/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_13/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_14/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_15/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_16/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_17/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_18/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_19/citeseer_gcn.pt',
    #  './repeat_target_models/repeat_20/citeseer_gcn.pt']

    repeat_path_mutation_model_list = ['./new_mutation_models/'+'repeat_'+str(i)+'/'+target_model_name+'/' for i in range(1, 21)]

    # ['./new_mutation_models/repeat_1/citeseer_gcn/',
    #  './new_mutation_models/repeat_2/citeseer_gcn/',
    #  './new_mutation_models/repeat_3/citeseer_gcn/',
    #  './new_mutation_models/repeat_4/citeseer_gcn/',
    #  './new_mutation_models/repeat_5/citeseer_gcn/',
    #  './new_mutation_models/repeat_6/citeseer_gcn/',
    #  './new_mutation_models/repeat_7/citeseer_gcn/',
    #  './new_mutation_models/repeat_8/citeseer_gcn/',
    #  './new_mutation_models/repeat_9/citeseer_gcn/',
    #  './new_mutation_models/repeat_10/citeseer_gcn/',
    #  './new_mutation_models/repeat_11/citeseer_gcn/',
    #  './new_mutation_models/repeat_12/citeseer_gcn/',
    #  './new_mutation_models/repeat_13/citeseer_gcn/',
    #  './new_mutation_models/repeat_14/citeseer_gcn/',
    #  './new_mutation_models/repeat_15/citeseer_gcn/',
    #  './new_mutation_models/repeat_16/citeseer_gcn/',
    #  './new_mutation_models/repeat_17/citeseer_gcn/',
    #  './new_mutation_models/repeat_18/citeseer_gcn/',
    #  './new_mutation_models/repeat_19/citeseer_gcn/',
    #  './new_mutation_models/repeat_20/citeseer_gcn/']



    target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, target_model_path)
    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()
    mutation_pre_idx_np = np.array([model(x, edge_index).argmax(dim=1).numpy() for model in model_list]).T
    feature_list = []
    for i in range(len(target_pre)):
        tmp_list = []
        for j in range(len(mutation_pre_idx_np[i])):
            if mutation_pre_idx_np[i][j] != target_pre[i]:
                tmp_list.append(1)
            else:
                tmp_list.append(0)
        feature_list.append(tmp_list)
    feature_np = np.array(feature_list)

    label_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != y[i]:
            label_list.append(1)
        else:
            label_list.append(0)
    label_np = np.array(label_list)

    return feature_np, label_np


def apfd(error_idx_list, pri_idx_list):
    error_idx_list = list(error_idx_list)
    pri_idx_list = list(pri_idx_list)
    n = len(pri_idx_list)
    m = len(error_idx_list)
    TF_list = [pri_idx_list.index(i) for i in error_idx_list]
    apfd = 1 - sum(TF_list)*1.0 / (n*m) + 1 / (2*n)
    return apfd


# edge_index_np.pkl   (2, 10556)
# [[   0    0    0 ... 2707 2707 2707]
#  [ 633 1862 2582 ...  598 1473 2706]]

# to

# modified_adj
# [[0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         ...,
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.]]


def edge_index_to_adj(edge_index_np):
    n_node = max(edge_index_np[0])+1
    m = np.full((n_node, n_node), 0)
    i_j_list = []
    for idx in range(len(edge_index_np[0])):
        i = edge_index_np[0][idx]
        j = edge_index_np[1][idx]
        if [i, j] not in i_j_list and [j, i] not in i_j_list:
            i_j_list.append([i, j])

    for v in i_j_list:
        i = v[0]
        j = v[1]
        m[i][j] = 1
        m[j][i] = 1
    return m


def adj_to_edge_index(adj):
    n_node = len(adj)
    up_list = []
    down_list = []
    for i in range(n_node):
        for j in range(n_node):
            if adj[i][j]==1:
                up_list.append(i)
                down_list.append(j)
                up_list.append(j)
                down_list.append(i)
    m = np.array([up_list, down_list])
    return m


def get_mutation_nodel_edge_features(num_node_features, target_hidden_channel, num_classes, target_model_path, x, y, edge_index, mutation_edge_index_list, model_name):
    target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, target_model_path)
    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()
    mutation_pre_idx_np = np.array([target_model(x, torch.from_numpy(tmp_edge_index)).argmax(dim=1).numpy() for tmp_edge_index in mutation_edge_index_list]).T

    feature_list = []
    for i in range(len(target_pre)):
        tmp_list = []
        for j in range(len(mutation_pre_idx_np[i])):
            if mutation_pre_idx_np[i][j] != target_pre[i]:
                tmp_list.append(1)
            else:
                tmp_list.append(0)
        feature_list.append(tmp_list)
    feature_np = np.array(feature_list)

    label_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != y[i]:
            label_list.append(1)
        else:
            label_list.append(0)
    label_np = np.array(label_list)

    return feature_np, label_np


def get_mutation_nodel_attribute_features(num_node_features, target_hidden_channel, num_classes, target_model_path, x, y, edge_index, mutation_attribute_index_list, model_name):
    target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, target_model_path)
    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()
    mutation_pre_idx_np = np.array([target_model(torch.from_numpy(tmp_x.astype(np.float32)), edge_index).argmax(dim=1).numpy() for tmp_x in mutation_attribute_index_list]).T

    feature_list = []
    for i in range(len(target_pre)):
        tmp_list = []
        for j in range(len(mutation_pre_idx_np[i])):
            if mutation_pre_idx_np[i][j] != target_pre[i]:
                tmp_list.append(1)
            else:
                tmp_list.append(0)
        feature_list.append(tmp_list)
    feature_np = np.array(feature_list)

    label_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != y[i]:
            label_list.append(1)
        else:
            label_list.append(0)
    label_np = np.array(label_list)

    return feature_np, label_np


def effect_size(x1, x2):
    mean1, mean2 = np.mean(x1), np.mean(x2)
    std1, std2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    pooled_std = np.sqrt(((len(x1)-1)*std1**2 + (len(x2)-1)*std2**2)/(len(x1)+len(x2)-2))
    return abs(mean1-mean2) / pooled_std


def p_value(x1, x2):
    t, p = ttest_ind(x1, x2)
    return p

