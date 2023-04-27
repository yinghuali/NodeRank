import numpy as np
import random
from scipy.special import softmax
from utils import *
from scipy.stats import entropy


def get_0_1_pro(pre_np):
    pre_np_0_1 = softmax(pre_np, axis=1)
    return pre_np_0_1


def Random_rank_idx(x):
    random_rank_idx = random.sample(range(0, len(x)), len(x))
    return random_rank_idx


def DeepGini_rank_idx(x):
    gini_score = 1 - np.sum(np.power(x, 2), axis=1)
    gini_rank_idx = gini_score.argsort()[::-1]
    return gini_rank_idx


def Mutation_rank_idx(num_node_features, target_hidden_channel, num_classes, target_model_path, x, edge_index, test_idx, model_list, model_name):
    target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, target_model_path)
    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()[test_idx]
    mutation_pre_idx_list = [model(x, edge_index).argmax(dim=1).numpy()[test_idx] for model in model_list]
    n_kill_model_np = np.array(get_n_kill_model(target_pre, mutation_pre_idx_list))
    select_rank_list = n_kill_model_np.argsort()[::-1]
    return select_rank_list


def Mutation_probability(num_node_features, target_hidden_channel, num_classes, target_model_path, x, edge_index, test_idx, model_list, model_name):
    target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, target_model_path)
    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()[test_idx]
    mutation_pre_idx_list = [model(x, edge_index).argmax(dim=1).numpy()[test_idx] for model in model_list]
    n_kill_model_np = np.array(get_n_kill_model(target_pre, mutation_pre_idx_list))
    n_kill_probability = n_kill_model_np / np.max(n_kill_model_np)
    return n_kill_probability


def VanillaSoftmax_rank_idx(x):
    value = 1 - x.max(1)
    vanillasoftmax_rank_idx = np.argsort(value)[::-1]
    return vanillasoftmax_rank_idx


def PCS_rank_idx(x):
    output_sort = np.sort(x)
    pcs_score = 1 - (output_sort[:, -1] - output_sort[:, -2])
    pcs_rank_idx = pcs_score.argsort()[::-1]
    return pcs_rank_idx


def Entropy_rank_idx(x):
    prob_dist = np.array([i / np.sum(i) for i in x])
    entropy_res = entropy(prob_dist, axis=1)
    entropy_rank_idx = np.argsort(entropy_res)[::-1]
    return entropy_rank_idx


