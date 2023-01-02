import numpy as np
import random
from scipy.special import softmax
from utils import *


def get_0_1_pro(pre_np):
    pre_np_0_1 = softmax(pre_np, axis=1)
    return pre_np_0_1


def Random_rank_idx(x):
    random_rank_idx = random.sample(range(0, len(x)), len(x))
    return random_rank_idx


def Margin_rank_idx(x):
    output_sort = np.sort(x)
    margin_score = output_sort[:, -1] - output_sort[:, -2]
    margin_rank_idx = margin_score.argsort()
    return margin_rank_idx


def DeepGini_rank_idx(x):
    gini_score = 1 - np.sum(np.power(x, 2), axis=1)
    gini_rank_idx = gini_score.argsort()[::-1]
    return gini_rank_idx


# def LeastConfidence_rank_idx(x):
#     max_pre = x.max(1)
#     leastConfidence_rank_idx = np.argsort(max_pre)
#     return leastConfidence_rank_idx


def MaximumProbability_rank_idx(x):
    max_pre = x.max(1)
    leastConfidence_rank_idx = np.argsort(max_pre)
    return leastConfidence_rank_idx


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


