import pickle
import torch
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack, Random, DICE, MinMax, PGDAttack, NodeEmbeddingAttack
from sklearn.model_selection import train_test_split
from scipy import sparse
from utils import *
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("--path_x_np", type=str)
ap.add_argument("--path_edge_index", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--save_edge_index", type=str)
args = ap.parse_args()

path_x_np = args.path_x_np
path_edge_index = args.path_edge_index
path_y = args.path_y
save_edge_index = args.save_edge_index


# python get_attack.py --path_x_np '../data/pubmed/x_np.pkl' --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y '../data/pubmed/y_np.pkl' --save_edge_index './test/attack_data/pubmed/pubmed'
# python get_attack.py --path_x_np '../data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y '../data/citeseer/y_np.pkl' --save_edge_index './test/attack_data/citeseer/citeseer'


def load_data(path_x_np, path_edge_index, path_y):

    x = pickle.load(open(path_x_np, 'rb'))
    edge_index = pickle.load(open(path_edge_index, 'rb'))
    adj = edge_index_to_adj(edge_index)
    y = pickle.load(open(path_y, 'rb'))

    idx_np = np.array(list(range(len(x))))
    train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.3, random_state=17)

    features = x
    labels = y
    adj = adj
    idx_train = train_idx
    idx_test = test_idx
    idx_val = None
    idx_unlabeled = test_idx

    print(x.shape)
    print(adj.shape)
    print(y.shape)

    return features, labels, adj, idx_train, idx_test, idx_val, idx_unlabeled


def get_dice(labels, adj, save_edge_index):
    model = DICE()
    adj_csr_matrix = sparse.csr_matrix(adj)
    model.attack(adj_csr_matrix, labels, n_perturbations=int(len(labels)*0.3))
    modified_adj = model.modified_adj
    modified_adj = modified_adj.A
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index+'_dice.pkl', 'wb'), protocol=4)


def get_minmax(surrogate, features, labels, adj, idx_train, idx_unlabeled, save_edge_index):
    model = MinMax(model=surrogate, nnodes=adj.shape[0], loss_type='CE', device='cuda').to('cuda')
    model.attack(features, adj, labels, idx_train, n_perturbations=int(len(labels)*0.3))
    modified_adj = model.modified_adj
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index+'_minmax.pkl', 'wb'), protocol=4)


def get_pgdattack(surrogate, features, labels, adj, idx_train, idx_unlabeled, save_edge_index):
    model = PGDAttack(model=surrogate, nnodes=adj.shape[0], loss_type='CE', device='cuda').to('cuda')
    model.attack(features, adj, labels, idx_train, n_perturbations=int(len(labels)*0.3))
    modified_adj = model.modified_adj
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index+'_pgdattack.pkl', 'wb'), protocol=4)


def get_randomattack_add(labels, adj, save_edge_index):
    model = Random()
    adj_csr_matrix = sparse.csr_matrix(adj)
    model.attack(adj_csr_matrix, type='add', n_perturbations=int(len(labels)*0.3))
    modified_adj = model.modified_adj
    modified_adj = modified_adj.A
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index+'_randomattack_add.pkl', 'wb'), protocol=4)


def get_randomattack_remove(labels, adj, save_edge_index):
    model = Random()
    adj_csr_matrix = sparse.csr_matrix(adj)
    model.attack(adj_csr_matrix, type='remove', n_perturbations=int(len(labels)*0.3))
    modified_adj = model.modified_adj
    modified_adj = modified_adj.A
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index+'_randomattack_remove.pkl', 'wb'), protocol=4)


def get_randomattack_flip(labels, adj, save_edge_index):
    model = Random()
    adj_csr_matrix = sparse.csr_matrix(adj)
    model.attack(adj_csr_matrix, type='flip', n_perturbations=int(len(labels)*0.3))
    modified_adj = model.modified_adj
    modified_adj = modified_adj.A
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index+'_randomattack_flip.pkl', 'wb'), protocol=4)


def get_nodeembeddingattack_add(adj, save_edge_index):
    model = NodeEmbeddingAttack()
    adj_csr_matrix = sparse.csr_matrix(adj)
    model.attack(adj_csr_matrix, attack_type="add", n_candidates=1000)
    modified_adj = model.modified_adj
    modified_adj = modified_adj.A
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index+'_nodeembeddingattack_add.pkl', 'wb'), protocol=4)


def get_nodeembeddingattack_remove(adj, save_edge_index):
    model = NodeEmbeddingAttack()
    adj_csr_matrix = sparse.csr_matrix(adj)
    model.attack(adj_csr_matrix, attack_type="remove", n_candidates=1000)
    modified_adj = model.modified_adj
    modified_adj = modified_adj.A
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index+'_nodeembeddingattack_remove.pkl', 'wb'), protocol=4)


def main():

    features, labels, adj, idx_train, idx_test, idx_val, idx_unlabeled = load_data(path_x_np, path_edge_index, path_y)

    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cuda').to('cuda')
    surrogate.fit(features, adj, labels, idx_train, patience=30)

    get_dice(labels, adj, save_edge_index)
    get_minmax(surrogate, features, labels, adj, idx_train, idx_unlabeled, save_edge_index)
    get_randomattack_add(labels, adj, save_edge_index)
    get_randomattack_remove(labels, adj, save_edge_index)
    get_randomattack_flip(labels, adj, save_edge_index)
    get_pgdattack(surrogate, features, labels, adj, idx_train, idx_unlabeled, save_edge_index)
    get_nodeembeddingattack_add(adj, save_edge_index)
    get_nodeembeddingattack_remove(adj, save_edge_index)


if __name__ == '__main__':
    main()









