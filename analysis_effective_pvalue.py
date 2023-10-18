import pandas as pd
import torch.nn.functional as F
import torch
from sklearn.linear_model import LogisticRegression
from get_rank_idx import *
from utils import *
import torch.utils.data as Data
from config import *
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--effectSize", type=float)
ap.add_argument("--model_name", type=str)
ap.add_argument("--subject_name", type=str)
ap.add_argument("--target_model_path", type=str)
ap.add_argument("--path_x_np", type=str)
ap.add_argument("--path_edge_index", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--path_mutation_edge_index_np_list", type=str)
ap.add_argument("--path_mutation_x_np_list", type=str)
args = ap.parse_args()

effectSize = args.effectSize
model_name = args.model_name
target_model_path = args.target_model_path
path_x_np = args.path_x_np
path_edge_index = args.path_edge_index
subject_name = args.subject_name
path_y = args.path_y
path_mutation_edge_index_np_list = args.path_mutation_edge_index_np_list
path_mutation_x_np_list = args.path_mutation_x_np_list


target_hidden_channel = 16

num_node_features, num_classes, x, edge_index, y, test_y, train_y, train_idx, test_idx = load_data(path_x_np, path_edge_index, path_y)

x_train = x[train_idx]
y_train = y[train_idx]

x_test = x[test_idx]
y_test = y[test_idx]


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


def get_repeat_mutation_model_features(subject_name):
    filtration_str = ['_dice', '_minmax', '_nodeembeddingattack_add', '_nodeembeddingattack_remove', '_pgdattack',
                      '_randomattack_add', '_randomattack_flip', '_randomattack_remove']
    new_subject_name = subject_name

    for i in filtration_str:
        new_subject_name = new_subject_name.replace(i, '')
    repeat_path_target_model_list = ['./repeat_target_models/'+'repeat_'+str(i)+'/'+new_subject_name+'.pt' for i in range(1, 21)]
    # ['./repeat_target_models/repeat_1/cora_gcn.pt', './repeat_target_models/repeat_2/cora_gcn.pt', './repeat_target_models/repeat_3/cora_gcn.pt', './repeat_target_models/repeat_4/cora_gcn.pt', './repeat_target_models/repeat_5/cora_gcn.pt', './repeat_target_models/repeat_6/cora_gcn.pt', './repeat_target_models/repeat_7/cora_gcn.pt', './repeat_target_models/repeat_8/cora_gcn.pt', './repeat_target_models/repeat_9/cora_gcn.pt', './repeat_target_models/repeat_10/cora_gcn.pt', './repeat_target_models/repeat_11/cora_gcn.pt', './repeat_target_models/repeat_12/cora_gcn.pt', './repeat_target_models/repeat_13/cora_gcn.pt', './repeat_target_models/repeat_14/cora_gcn.pt', './repeat_target_models/repeat_15/cora_gcn.pt', './repeat_target_models/repeat_16/cora_gcn.pt', './repeat_target_models/repeat_17/cora_gcn.pt', './repeat_target_models/repeat_18/cora_gcn.pt', './repeat_target_models/repeat_19/cora_gcn.pt', './repeat_target_models/repeat_20/cora_gcn.pt']

    repeat_path_mutation_model_list = ['./new_mutation_models/' + 'repeat_' + str(i) + '/' + new_subject_name + '/' for i in range(1, 21)]
    # ['./new_mutation_gitmodels/repeat_1/cora_gcn/', './new_mutation_models/repeat_2/cora_gcn/', './new_mutation_models/repeat_3/cora_gcn/', './new_mutation_models/repeat_4/cora_gcn/', './new_mutation_models/repeat_5/cora_gcn/', './new_mutation_models/repeat_6/cora_gcn/', './new_mutation_models/repeat_7/cora_gcn/', './new_mutation_models/repeat_8/cora_gcn/', './new_mutation_models/repeat_9/cora_gcn/', './new_mutation_models/repeat_10/cora_gcn/', './new_mutation_models/repeat_11/cora_gcn/', './new_mutation_models/repeat_12/cora_gcn/', './new_mutation_models/repeat_13/cora_gcn/', './new_mutation_models/repeat_14/cora_gcn/', './new_mutation_models/repeat_15/cora_gcn/', './new_mutation_models/repeat_16/cora_gcn/', './new_mutation_models/repeat_17/cora_gcn/', './new_mutation_models/repeat_18/cora_gcn/', './new_mutation_models/repeat_19/cora_gcn/', './new_mutation_models/repeat_20/cora_gcn/']

    repeat_target_model_list = []
    for model_path in repeat_path_target_model_list:
        target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, model_path)
        repeat_target_model_list.append(target_model)

    repeat_mutation_model_list = []
    for path_mutation_model in repeat_path_mutation_model_list:
        mutation_model_list = []
        path_model_list = get_model_path(path_mutation_model)
        path_model_list = sorted(path_model_list)
        path_config_list = [i.replace('.pt', '.pkl') for i in path_model_list]
        hidden_channel_list = [int(i.split('/')[-1].split('_')[2]) for i in path_config_list]
        dic_list = [pickle.load(open(i, 'rb')) for i in path_config_list]

        for i in range(len(path_model_list)):
            try:
                tmp_model = load_model(model_name, path_model_list[i], hidden_channel_list[i], num_node_features, num_classes, dic_list[i])
                mutation_model_list.append(tmp_model)
            except:
                print(dic_list[i])
        repeat_mutation_model_list.append(mutation_model_list)

    repeat_target_pre_list = []
    repeat_mutation_pre_idx_np_list = []
    for i in range(len(repeat_path_target_model_list)):
        target_model = repeat_target_model_list[i]
        model_list = repeat_mutation_model_list[i]
        target_model.eval()

        target_pre = target_model(x, edge_index).argmax(dim=1).numpy()
        mutation_pre_idx_np = np.array([model(x, edge_index).argmax(dim=1).numpy() for model in model_list]).T
        repeat_target_pre_list.append(target_pre)
        repeat_mutation_pre_idx_np_list.append(mutation_pre_idx_np)

    mutation_model_feaure = []
    for i in range(len(x)):
        tmp_feaure = []
        repeat_target_pre = [repeat_target_pre_list[r][i] for r in range(20)]
        repaet_mutation_pre_idx_np = [repeat_mutation_pre_idx_np_list[r][i] for r in range(20)]

        n_mutants = len(repaet_mutation_pre_idx_np[0])
        for j in range(n_mutants):
            repeat_mutants_pre = [repaet_mutation_pre_idx_np[r][j] for r in range(20)]
            if str(repeat_target_pre) == str(repeat_mutants_pre):
                tmp_feaure.append(0)
            elif len(set(repeat_target_pre)) == 1 and len(set(repeat_mutants_pre)) == 1:
                tmp_feaure.append(1)
            else:
                es = effect_size(repeat_target_pre, repeat_mutants_pre)
                pvalue = p_value(repeat_target_pre, repeat_mutants_pre)

                if es >= effectSize and pvalue < 0.05:
                    tmp_feaure.append(1)
                else:
                    tmp_feaure.append(0)

        mutation_model_feaure.append(tmp_feaure)
    feature_np = np.array(mutation_model_feaure)

    target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, target_model_path)
    target_model.eval()
    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()

    label_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != y[i]:
            label_list.append(1)
        else:
            label_list.append(0)
    label_np = np.array(label_list)
    return feature_np, label_np

# mutant model feature
feature_model_np, label_model_np = get_repeat_mutation_model_features(subject_name)

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
    target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, target_model_path)
    target_model.eval()
    # XGB
    model = XGBClassifier()
    model.fit(x_train, y_train)
    xgb_pre = model.predict(x_test)
    y_pred_train_xgb = model.predict_proba(x_train)[:, 1]
    y_pred_test_xgb = model.predict_proba(x_test)[:, 1]
    xgb_rank_idx = y_pred_test_xgb.argsort()[::-1].copy()
    xgb_rank_idx_train = y_pred_train_xgb.argsort()[::-1].copy()

    # LR
    model = LogisticRegression(solver='liblinear')
    model.fit(x_train, y_train)
    lr_pre = model.predict(x_test)
    y_pred_train_lr = model.predict_proba(x_train)[:, 1]
    y_pred_test_lr = model.predict_proba(x_test)[:, 1]
    lr_rank_idx = y_pred_test_lr.argsort()[::-1].copy()
    lr_rank_idx_train = y_pred_train_lr.argsort()[::-1].copy()

    # RF
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    rf_pre = model.predict(x_test)
    y_pred_train_rf = model.predict_proba(x_train)[:, 1]
    y_pred_test_rf = model.predict_proba(x_test)[:, 1]
    rf_rank_idx = y_pred_test_rf.argsort()[::-1].copy()
    rf_rank_idx_train = y_pred_train_rf.argsort()[::-1].copy()

    # LGBM
    model = LGBMClassifier()
    model.fit(x_train, y_train)
    lgb_pre = model.predict(x_test)
    y_pred_train_lgb = model.predict_proba(x_train)[:, 1]
    y_pred_test_lgb = model.predict_proba(x_test)[:, 1]
    lgb_rank_idx = y_pred_test_lgb.argsort()[::-1].copy()
    lgb_rank_idx_train = y_pred_train_lgb.argsort()[::-1].copy()

    # fusion model
    y_pred_test_fusion = y_pred_test_xgb + y_pred_test_lr + y_pred_test_rf + y_pred_test_lgb
    fusion_rank_idx = y_pred_test_fusion.argsort()[::-1].copy()

    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()[test_idx]
    idx_miss_list = get_idx_miss_class(target_pre, test_y)
    x_test_target_model_pre = target_model(x, edge_index).detach().numpy()[test_idx]

    target_pre_train = target_model(x, edge_index).argmax(dim=1).numpy()[train_idx]
    idx_miss_list_train = get_idx_miss_class(target_pre_train, train_y)

    fusion_apfd = apfd(idx_miss_list, fusion_rank_idx)

    content = subject_name+'->'+str(effectSize)+'->'+str(fusion_apfd)
    write_result(content, 'results_analysis_effective_pvalue.txt')


if __name__ == '__main__':
    main()

