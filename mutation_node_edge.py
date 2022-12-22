import pickle
import numpy as np
import random
import argparse
from utils import adj_to_edge_index, edge_index_to_adj

ap = argparse.ArgumentParser()
ap.add_argument("--path_edge_index_np", type=str)
ap.add_argument("--n_edges", type=int)
ap.add_argument("--save_path_pkl", type=str)
args = ap.parse_args()

path_edge_index_np = args.path_edge_index_np
n_edges = args.n_edges
save_path_pkl = args.save_path_pkl

# python mutation_node_edge.py --path_edge_index_np 'data/cora/edge_index_np.pkl' --n_edges 480 --save_path_pkl 'data/cora/mutation_edge_index_np_list.pkl'


def get_mutation_node_edge(path_edge_index_np, n_edges, save_path_pkl):
    data_list = []
    edge_index_np = pickle.load(open(path_edge_index_np, 'rb'))
    adj = edge_index_to_adj(edge_index_np)
    idx_list = list(range(len(adj)))

    for i in range(n_edges):
        tmp_adj = np.copy(adj)
        for j in range(len(tmp_adj)):
            mutation_idx_list = random.sample(idx_list, 1)
            for k in mutation_idx_list:
                tmp_adj[j][k] = 1
        tmp_edge_index = adj_to_edge_index(tmp_adj)
        data_list.append(tmp_edge_index)
        print('=====', i, '========')
    pickle.dump(data_list, open(save_path_pkl, 'wb'), protocol=4)


get_mutation_node_edge(path_edge_index_np, n_edges, save_path_pkl)


