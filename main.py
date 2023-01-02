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


# path_model_file = './mutation_models/cora_gcn'
# model_name = 'gcn'
# target_model_path = './target_models/cora_gcn.pt'
# path_x_np = './data/cora/x_np.pkl'
# path_edge_index = './data/cora/edge_index_np.pkl'
# path_y = './data/cora/y_np.pkl'
# subject_name = 'cora_gcn'
# path_mutation_edge_index_np_list = './data/cora/mutation_edge_index_np_list.pkl'
# path_mutation_x_np_list = './data/cora/mutation_x_np_list.pkl'


target_hidden_channel = 16
path_result_pfd = 'res/pfd' + '_' + subject_name + '.csv'
path_result_apfd = 'res/apfd' + '_' + subject_name + '.csv'

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

    # XGB
    model = XGBClassifier()
    model.fit(x_train, y_train)
    xgb_pre = model.predict(x_test)
    y_pred_train_xgb = model.predict_proba(x_train)[:, 1]
    y_pred_test_xgb = model.predict_proba(x_test)[:, 1]
    xgb_rank_idx = y_pred_test_xgb.argsort()[::-1].copy()

    # LR
    model = LogisticRegression(solver='liblinear')
    model.fit(x_train, y_train)
    lr_pre = model.predict(x_test)
    y_pred_train_lr = model.predict_proba(x_train)[:, 1]
    y_pred_test_lr = model.predict_proba(x_test)[:, 1]
    lr_rank_idx = y_pred_test_lr.argsort()[::-1].copy()

    # RF
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    rf_pre = model.predict(x_test)
    y_pred_train_rf = model.predict_proba(x_train)[:, 1]
    y_pred_test_rf = model.predict_proba(x_test)[:, 1]
    rf_rank_idx = y_pred_test_rf.argsort()[::-1].copy()

    # LGBM
    model = LGBMClassifier()
    model.fit(x_train, y_train)
    lgb_pre = model.predict(x_test)
    y_pred_train_lgb = model.predict_proba(x_train)[:, 1]
    y_pred_test_lgb = model.predict_proba(x_test)[:, 1]
    lgb_rank_idx = y_pred_test_lgb.argsort()[::-1].copy()

    # fusion model
    y_pred_test_fusion = y_pred_test_xgb + y_pred_test_lr + y_pred_test_rf + y_pred_test_lgb

    fusion_rank_idx = y_pred_test_fusion.argsort()[::-1].copy()

    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()[test_idx]
    idx_miss_list = get_idx_miss_class(target_pre, test_y)

    mutation_rank_idx = Mutation_rank_idx(num_node_features, target_hidden_channel, num_classes, target_model_path, x,
                                          edge_index, test_idx, model_list, model_name)

    x_test_target_model_pre = target_model(x, edge_index).detach().numpy()[test_idx]

    margin_rank_idx = Margin_rank_idx(x_test_target_model_pre)
    deepGini_rank_idx = DeepGini_rank_idx(x_test_target_model_pre)
    leastConfidence_rank_idx = MaximumProbability_rank_idx(x_test_target_model_pre)
    random_rank_idx = Random_rank_idx(x_test_target_model_pre)

    fusion_ratio_list = get_res_ratio_list(idx_miss_list, fusion_rank_idx, select_ratio_list)
    rf_ratio_list = get_res_ratio_list(idx_miss_list, rf_rank_idx, select_ratio_list)
    xgb_ratio_list = get_res_ratio_list(idx_miss_list, xgb_rank_idx, select_ratio_list)
    lgb_ratio_list = get_res_ratio_list(idx_miss_list, lgb_rank_idx, select_ratio_list)
    lr_ratio_list = get_res_ratio_list(idx_miss_list, lr_rank_idx, select_ratio_list)
    mutation_ratio_list = get_res_ratio_list(idx_miss_list, mutation_rank_idx, select_ratio_list)
    margin_ratio_list = get_res_ratio_list(idx_miss_list, margin_rank_idx, select_ratio_list)
    deepGini_ratio_list = get_res_ratio_list(idx_miss_list, deepGini_rank_idx, select_ratio_list)
    leastConfidence_ratio_list = get_res_ratio_list(idx_miss_list, leastConfidence_rank_idx, select_ratio_list)
    random_ratio_list = get_res_ratio_list(idx_miss_list, random_rank_idx, select_ratio_list)

    fusion_ratio_list.insert(0, subject_name + '_' + 'fusion')
    rf_ratio_list.insert(0, subject_name + '_' + 'rf')
    xgb_ratio_list.insert(0, subject_name+'_'+'xgb')
    lgb_ratio_list.insert(0, subject_name + '_' + 'lgb')
    lr_ratio_list.insert(0, subject_name+'_'+'lr')
    mutation_ratio_list.insert(0, subject_name+'_'+'mutation')
    margin_ratio_list.insert(0, subject_name+'_'+'margin')
    deepGini_ratio_list.insert(0, subject_name+'_'+'deepGini')
    leastConfidence_ratio_list.insert(0, subject_name+'_'+'leastConfidence')
    random_ratio_list.insert(0, subject_name+'_'+'random')

    res_list = [xgb_ratio_list, lgb_ratio_list, rf_ratio_list, lr_ratio_list, mutation_ratio_list, fusion_ratio_list, margin_ratio_list, deepGini_ratio_list, leastConfidence_ratio_list, random_ratio_list]
    df = pd.DataFrame(columns=None, data=res_list)
    df.to_csv(path_result_pfd, mode='a', header=False, index=False)

    fusion_apfd = [apfd(idx_miss_list, fusion_rank_idx)]
    rf_apfd = [apfd(idx_miss_list, rf_rank_idx)]
    xgb_apfd = [apfd(idx_miss_list, xgb_rank_idx)]
    lgb_apfd = [apfd(idx_miss_list, lgb_rank_idx)]
    mutation_apfd = [apfd(idx_miss_list, mutation_rank_idx)]
    lr_apfd = [apfd(idx_miss_list, lr_rank_idx)]
    deepGini_apfd = [apfd(idx_miss_list, deepGini_rank_idx)]
    leastConfidence_apfd = [apfd(idx_miss_list, leastConfidence_rank_idx)]
    margin_apfd = [apfd(idx_miss_list, margin_rank_idx)]
    random_apfd = [apfd(idx_miss_list, random_rank_idx)]

    fusion_apfd.insert(0, subject_name + '_' + 'fusion')
    rf_apfd.insert(0, subject_name + '_' + 'rf')
    xgb_apfd.insert(0, subject_name+'_'+'xgb')
    lgb_apfd.insert(0, subject_name + '_' + 'lgb')
    lr_apfd.insert(0, subject_name+'_'+'lr')
    mutation_apfd.insert(0, subject_name+'_'+'mutation')
    margin_apfd.insert(0, subject_name+'_'+'margin')
    deepGini_apfd.insert(0, subject_name+'_'+'deepGini')
    leastConfidence_apfd.insert(0, subject_name+'_'+'leastConfidence')
    random_apfd.insert(0, subject_name+'_'+'random')

    res_list = [xgb_apfd, lgb_apfd, rf_apfd, lr_apfd, mutation_apfd, fusion_apfd, margin_apfd, deepGini_apfd, leastConfidence_apfd, random_apfd]
    df = pd.DataFrame(columns=None, data=res_list)
    df.to_csv(path_result_apfd, mode='a', header=False, index=False)

    # Different model fusion methods
    y_pred_test_fusion_weight = y_pred_test_xgb*apfd(idx_miss_list, xgb_rank_idx) + y_pred_test_lr*apfd(idx_miss_list, lr_rank_idx) + \
                                y_pred_test_rf*apfd(idx_miss_list, rf_rank_idx) + y_pred_test_lgb*apfd(idx_miss_list, lgb_rank_idx)
    fusion_weight_rank_idx = y_pred_test_fusion_weight.argsort()[::-1].copy()
    fusion_weight_ratio_list = get_res_ratio_list(idx_miss_list, fusion_weight_rank_idx, select_ratio_list)
    fusion_weight_ratio_list.insert(0, subject_name + '_' + 'fusion_weight')

    res_list = [fusion_weight_ratio_list]
    df = pd.DataFrame(columns=None, data=res_list)
    df.to_csv(path_result_pfd, mode='a', header=False, index=False)

    fusion_weight_apfd = [apfd(idx_miss_list, fusion_weight_rank_idx)]
    fusion_weight_apfd.insert(0, subject_name + '_' + 'fusion_weight')

    res_list = [fusion_weight_apfd]
    df = pd.DataFrame(columns=None, data=res_list)
    df.to_csv(path_result_apfd, mode='a', header=False, index=False)

    # Stacking
    feature_stacking_train_pre = np.array([y_pred_train_xgb, y_pred_train_lr, y_pred_train_rf, y_pred_train_lgb]).T
    feature_stacking_test_pre = np.array([y_pred_test_xgb, y_pred_test_lr, y_pred_test_rf, y_pred_test_lgb]).T
    model = RandomForestClassifier()
    model.fit(feature_stacking_train_pre, y_train)
    y_pred_test_stacking = model.predict_proba(feature_stacking_test_pre)[:, 1]
    stacking_rank_idx = y_pred_test_stacking.argsort()[::-1].copy()
    fusion_stacking_ratio_list = get_res_ratio_list(idx_miss_list, stacking_rank_idx, select_ratio_list)
    fusion_stacking_ratio_list.insert(0, subject_name + '_' + 'fusion_stacking')

    res_list = [fusion_stacking_ratio_list]
    df = pd.DataFrame(columns=None, data=res_list)
    df.to_csv(path_result_pfd, mode='a', header=False, index=False)

    fusion_stacking_apfd = [apfd(idx_miss_list, stacking_rank_idx)]
    fusion_stacking_apfd.insert(0, subject_name + '_' + 'fusion_stacking')

    res_list = [fusion_stacking_apfd]
    df = pd.DataFrame(columns=None, data=res_list)
    df.to_csv(path_result_apfd, mode='a', header=False, index=False)

    # voting
    voting_pre = xgb_pre+lr_pre+rf_pre+lgb_pre
    voting_rank_idx = voting_pre.argsort()[::-1].copy()
    fusion_voting_ratio_list = get_res_ratio_list(idx_miss_list, voting_rank_idx, select_ratio_list)
    fusion_voting_ratio_list.insert(0, subject_name + '_' + 'fusion_voting')

    res_list = [fusion_voting_ratio_list]
    df = pd.DataFrame(columns=None, data=res_list)
    df.to_csv(path_result_pfd, mode='a', header=False, index=False)

    fusion_voting_apfd = [apfd(idx_miss_list, voting_rank_idx)]
    fusion_voting_apfd.insert(0, subject_name + '_' + 'fusion_voting')

    res_list = [fusion_voting_apfd]
    df = pd.DataFrame(columns=None, data=res_list)
    df.to_csv(path_result_apfd, mode='a', header=False, index=False)


if __name__ == '__main__':
    main()


