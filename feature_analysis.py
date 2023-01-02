import pandas as pd
import torch.nn.functional as F
import json
import torch
from sklearn.linear_model import LogisticRegression
from get_rank_idx import *
from utils import *
import torch.utils.data as Data
from config import *
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn import metrics

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--path_model_file", type=str)
ap.add_argument("--model_name", type=str)
ap.add_argument("--target_model_path", type=str)
ap.add_argument("--path_x_np", type=str)
ap.add_argument("--path_edge_index", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--subject_name", type=str)
ap.add_argument("--path_mutation_edge_index_np_list", type=str)
ap.add_argument("--path_mutation_x_np_list", type=str)
args = ap.parse_args()

# python main.py --path_model_file './mutation_models/cora_gcn' --model_name 'gcn' --target_model_path './target_models/cora_gcn.pt' --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --subject_name 'cora_gcn' --path_mutation_edge_index_np_list './data/cora/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/cora/mutation_x_np_list.pkl'

path_model_file = args.path_model_file
model_name = args.model_name
target_model_path = args.target_model_path
path_x_np = args.path_x_np
path_edge_index = args.path_edge_index
path_y = args.path_y
subject_name = args.subject_name
path_mutation_edge_index_np_list = args.path_mutation_edge_index_np_list
path_mutation_x_np_list = args.path_mutation_x_np_list


# path_model_file = './mutation_models/citeseer_gcn'
# model_name = 'gcn'
# target_model_path = './target_models/citeseer_gcn.pt'
# path_x_np = './data/citeseer/x_np.pkl'
# path_edge_index = './data/citeseer/edge_index_np.pkl'
# path_y = './data/citeseer/y_np.pkl'
# subject_name = 'citeseer_gcn'
# path_mutation_edge_index_np_list = './data/citeseer/mutation_edge_index_np_list.pkl'
# path_mutation_x_np_list = './data/citeseer/mutation_x_np_list.pkl'

target_hidden_channel = 16
path_result_pfd = 'res/features/pfd' + '_' + subject_name + '.csv'
path_result_apfd = 'res/features/apfd' + '_' + subject_name + '.csv'
num_node_features, num_classes, x, edge_index, y, test_y, train_idx, test_idx = load_data(path_x_np, path_edge_index, path_y)
path_model_list = get_model_path(path_model_file)
path_model_list = sorted(path_model_list)
path_config_list = [i.replace('.pt', '.pkl') for i in path_model_list]
hidden_channel_list = [int(i.split('/')[-1].split('_')[2]) for i in path_config_list]
dic_list = [pickle.load(open(i, 'rb')) for i in path_config_list]


model_list = []
for i in range(len(path_model_list)):
    try:
        tmp_model = load_model(model_name, path_model_list[i], hidden_channel_list[i], num_node_features, num_classes, dic_list[i])
        model_list.append(tmp_model)
    except:
        print(dic_list[i])

print('number of models:', len(path_model_list))
print('number of models loaded:', len(model_list))

target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, target_model_path)


# mutant model feature
feature_model_np, label_model_np = get_mutation_model_features(num_node_features, target_hidden_channel, num_classes, target_model_path, x, y, edge_index, model_list, model_name)

# mutant node edge feature
mutation_edge_index_list = pickle.load(open(path_mutation_edge_index_np_list, 'rb'))
feature_node_edge_np, label_node_edge_np = get_mutation_nodel_edge_features(num_node_features, target_hidden_channel, num_classes, target_model_path, x, y, edge_index, mutation_edge_index_list, model_name)

# mutant node attribute feature
mutation_attribute_index_list = pickle.load(open(path_mutation_x_np_list, 'rb'))
feature_node_attribute_np, label_node_attribute_np = get_mutation_nodel_attribute_features(num_node_features, target_hidden_channel, num_classes, target_model_path, x, y, edge_index, mutation_attribute_index_list, model_name)

feature_np = np.concatenate((feature_model_np, feature_node_edge_np, feature_node_attribute_np), axis=1)
label_np = label_model_np

x_train = feature_np[train_idx]
y_train = label_np[train_idx]
x_test = feature_np[test_idx]
y_test = label_np[test_idx]


def main():

    # one feature
    x_train = feature_node_attribute_np[train_idx]
    x_test = feature_node_attribute_np[test_idx]

    model = XGBClassifier()
    model.fit(x_train, y_train)
    y_pred_test_xgb = model.predict_proba(x_test)[:, 1]

    model = LogisticRegression(solver='liblinear')
    model.fit(x_train, y_train)
    y_pred_test_lr = model.predict_proba(x_test)[:, 1]

    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred_test_rf = model.predict_proba(x_test)[:, 1]

    model = LGBMClassifier()
    model.fit(x_train, y_train)
    y_pred_test_lgb = model.predict_proba(x_test)[:, 1]

    y_pred_test_fusion = y_pred_test_xgb + y_pred_test_lr + y_pred_test_rf + y_pred_test_lgb
    fusion_rank_idx = y_pred_test_fusion.argsort()[::-1].copy()

    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()[test_idx]
    idx_miss_list = get_idx_miss_class(target_pre, test_y)
    one_feature_apfd = apfd(idx_miss_list, fusion_rank_idx)

    # two feature
    x_train = np.concatenate((feature_node_edge_np, feature_node_attribute_np), axis=1)[train_idx]
    x_test = np.concatenate((feature_node_edge_np, feature_node_attribute_np), axis=1)[test_idx]

    model = XGBClassifier()
    model.fit(x_train, y_train)
    y_pred_test_xgb = model.predict_proba(x_test)[:, 1]

    model = LogisticRegression(solver='liblinear')
    model.fit(x_train, y_train)
    y_pred_test_lr = model.predict_proba(x_test)[:, 1]

    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred_test_rf = model.predict_proba(x_test)[:, 1]

    model = LGBMClassifier()
    model.fit(x_train, y_train)
    y_pred_test_lgb = model.predict_proba(x_test)[:, 1]

    y_pred_test_fusion = y_pred_test_xgb + y_pred_test_lr + y_pred_test_rf + y_pred_test_lgb
    fusion_rank_idx = y_pred_test_fusion.argsort()[::-1].copy()

    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()[test_idx]
    idx_miss_list = get_idx_miss_class(target_pre, test_y)
    two_feature_apfd = apfd(idx_miss_list, fusion_rank_idx)

    # three feature
    x_train = np.concatenate((feature_model_np, feature_node_edge_np, feature_node_attribute_np), axis=1)[train_idx]
    x_test = np.concatenate((feature_model_np, feature_node_edge_np, feature_node_attribute_np), axis=1)[test_idx]

    model = XGBClassifier()
    model.fit(x_train, y_train)
    y_pred_test_xgb = model.predict_proba(x_test)[:, 1]

    model = LogisticRegression(solver='liblinear')
    model.fit(x_train, y_train)
    y_pred_test_lr = model.predict_proba(x_test)[:, 1]

    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred_test_rf = model.predict_proba(x_test)[:, 1]

    model = LGBMClassifier()
    model.fit(x_train, y_train)
    y_pred_test_lgb = model.predict_proba(x_test)[:, 1]

    y_pred_test_fusion = y_pred_test_xgb + y_pred_test_lr + y_pred_test_rf + y_pred_test_lgb
    fusion_rank_idx = y_pred_test_fusion.argsort()[::-1].copy()

    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()[test_idx]
    idx_miss_list = get_idx_miss_class(target_pre, test_y)
    three_feature_apfd = apfd(idx_miss_list, fusion_rank_idx)

    res_list = [one_feature_apfd, two_feature_apfd, three_feature_apfd]
    df = pd.DataFrame(columns=None, data=res_list)
    df.to_csv(path_result_apfd, mode='a', header=False, index=False)

main()



