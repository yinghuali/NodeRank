import pickle
import numpy as np
import random
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("--path_node_feature", type=str)
ap.add_argument("--n_feature", type=int)
ap.add_argument("--save_path_pkl", type=str)
args = ap.parse_args()

# python parameter_mutation_node_feature.py --path_node_feature 'data/cora/x_np.pkl' --n_feature 50 --save_path_pkl 'data/cora/mutation_x_np_list.pkl'


path_node_feature = args.path_node_feature
n_feature = args.n_feature
save_path_pkl = args.save_path_pkl

save_path_pkl_5 = save_path_pkl.replace('.pkl', '_5.pkl')
save_path_pkl_10 = save_path_pkl.replace('.pkl', '_10.pkl')
save_path_pkl_15 = save_path_pkl.replace('.pkl', '_15.pkl')
save_path_pkl_20 = save_path_pkl.replace('.pkl', '_20.pkl')


def get_mutation_node_feature_05(path_node_feature, n_feature, save_path_pkl):
    data_list = []
    n_g_feature = int(n_feature/5)
    x_np = pickle.load(open(path_node_feature, 'rb'))
    row = x_np.shape[0]
    col = x_np.shape[1]

    for i in range(n_g_feature):
        tmp_x = np.around(np.random.rand(row, col) * 0.05, 2)
        data_list.append(x_np+tmp_x)

        tmp_x = np.around(np.random.rand(row, col) * 0.05, 2)
        data_list.append(x_np+tmp_x)

        tmp_x = np.around(np.random.rand(row, col) * 0.05, 2)
        data_list.append(x_np+tmp_x)

        tmp_x = np.around(np.random.rand(row, col) * 0.05, 2)
        data_list.append(x_np+tmp_x)

        tmp_x = np.around(np.random.rand(row, col) * 0.05, 2)
        data_list.append(x_np+tmp_x)

    pickle.dump(data_list, open(save_path_pkl, 'wb'), protocol=4)


def get_mutation_node_feature_10(path_node_feature, n_feature, save_path_pkl):
    data_list = []
    n_g_feature = int(n_feature/5)
    x_np = pickle.load(open(path_node_feature, 'rb'))
    row = x_np.shape[0]
    col = x_np.shape[1]

    for i in range(n_g_feature):
        tmp_x = np.around(np.random.rand(row, col) * 0.10, 2)
        data_list.append(x_np+tmp_x)

        tmp_x = np.around(np.random.rand(row, col) * 0.10, 2)
        data_list.append(x_np+tmp_x)

        tmp_x = np.around(np.random.rand(row, col) * 0.10, 2)
        data_list.append(x_np+tmp_x)

        tmp_x = np.around(np.random.rand(row, col) * 0.10, 2)
        data_list.append(x_np+tmp_x)

        tmp_x = np.around(np.random.rand(row, col) * 0.10, 2)
        data_list.append(x_np+tmp_x)

    pickle.dump(data_list, open(save_path_pkl, 'wb'), protocol=4)


def get_mutation_node_feature_15(path_node_feature, n_feature, save_path_pkl):
    data_list = []
    n_g_feature = int(n_feature/5)
    x_np = pickle.load(open(path_node_feature, 'rb'))
    row = x_np.shape[0]
    col = x_np.shape[1]

    for i in range(n_g_feature):
        tmp_x = np.around(np.random.rand(row, col) * 0.15, 2)
        data_list.append(x_np+tmp_x)

        tmp_x = np.around(np.random.rand(row, col) * 0.15, 2)
        data_list.append(x_np+tmp_x)

        tmp_x = np.around(np.random.rand(row, col) * 0.15, 2)
        data_list.append(x_np+tmp_x)

        tmp_x = np.around(np.random.rand(row, col) * 0.15, 2)
        data_list.append(x_np+tmp_x)

        tmp_x = np.around(np.random.rand(row, col) * 0.15, 2)
        data_list.append(x_np+tmp_x)

    pickle.dump(data_list, open(save_path_pkl, 'wb'), protocol=4)


def get_mutation_node_feature_20(path_node_feature, n_feature, save_path_pkl):
    data_list = []
    n_g_feature = int(n_feature/5)
    x_np = pickle.load(open(path_node_feature, 'rb'))
    row = x_np.shape[0]
    col = x_np.shape[1]

    for i in range(n_g_feature):
        tmp_x = np.around(np.random.rand(row, col) * 0.20, 2)
        data_list.append(x_np+tmp_x)

        tmp_x = np.around(np.random.rand(row, col) * 0.20, 2)
        data_list.append(x_np+tmp_x)

        tmp_x = np.around(np.random.rand(row, col) * 0.20, 2)
        data_list.append(x_np+tmp_x)

        tmp_x = np.around(np.random.rand(row, col) * 0.20, 2)
        data_list.append(x_np+tmp_x)

        tmp_x = np.around(np.random.rand(row, col) * 0.20, 2)
        data_list.append(x_np+tmp_x)

    pickle.dump(data_list, open(save_path_pkl, 'wb'), protocol=4)


get_mutation_node_feature_05(path_node_feature, n_feature, save_path_pkl_5)
get_mutation_node_feature_10(path_node_feature, n_feature, save_path_pkl_10)
get_mutation_node_feature_15(path_node_feature, n_feature, save_path_pkl_15)
get_mutation_node_feature_20(path_node_feature, n_feature, save_path_pkl_20)

