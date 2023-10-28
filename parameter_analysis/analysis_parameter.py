import pandas as pd
import torch.nn.functional as F
import torch
from sklearn.linear_model import LogisticRegression
from get_rank_idx import *
from utils import *
import torch.utils.data as Data
from parameter_config import *
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
ap.add_argument("--save_name", type=str)
args = ap.parse_args()

path_model_file = args.path_model_file
model_name = args.model_name
target_model_path = args.target_model_path
path_x_np = args.path_x_np
path_edge_index = args.path_edge_index
path_y = args.path_y
subject_name = args.subject_name
path_mutation_edge_index_np_list = args.path_mutation_edge_index_np_list
path_mutation_x_np_list = args.path_mutation_x_np_list
save_name = args.save_name

target_hidden_channel = 16
path_result_pfd = 'parameter_results/pfd' + '_' + subject_name + '_' + save_name + '.csv'
path_result_apfd = 'parameter_results/apfd' + '_' + subject_name + '_' + save_name + '.csv'

num_node_features, num_classes, x, edge_index, y, test_y, train_y, train_idx, test_idx = load_data(path_x_np, path_edge_index, path_y)
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

    deepGini_rank_idx = DeepGini_rank_idx(x_test_target_model_pre)
    random_rank_idx = Random_rank_idx(x_test_target_model_pre)
    vanillasm_rank_idx = VanillaSoftmax_rank_idx(x_test_target_model_pre)
    pcs_rank_idx = PCS_rank_idx(x_test_target_model_pre)
    entropy_rank_idx = Entropy_rank_idx(x_test_target_model_pre)

    fusion_ratio_list = get_res_ratio_list(idx_miss_list, fusion_rank_idx, select_ratio_list)
    deepGini_ratio_list = get_res_ratio_list(idx_miss_list, deepGini_rank_idx, select_ratio_list)
    random_ratio_list = get_res_ratio_list(idx_miss_list, random_rank_idx, select_ratio_list)
    vanillasm_ratio_list = get_res_ratio_list(idx_miss_list, vanillasm_rank_idx, select_ratio_list)
    pcs_ratio_list = get_res_ratio_list(idx_miss_list, pcs_rank_idx, select_ratio_list)
    entropy_ratio_list = get_res_ratio_list(idx_miss_list, entropy_rank_idx, select_ratio_list)

    fusion_ratio_list.insert(0, subject_name + '_' + 'fusion')
    deepGini_ratio_list.insert(0, subject_name+'_'+'deepGini')
    vanillasm_ratio_list.insert(0, subject_name+'_'+'vanillasm')
    pcs_ratio_list.insert(0, subject_name + '_' + 'pcs')
    entropy_ratio_list.insert(0, subject_name + '_' + 'entropy')
    random_ratio_list.insert(0, subject_name+'_'+'random')

    res_list = [fusion_ratio_list, deepGini_ratio_list, vanillasm_ratio_list, pcs_ratio_list, entropy_ratio_list, random_ratio_list]
    df = pd.DataFrame(columns=None, data=res_list)
    df.to_csv(path_result_pfd, mode='a', header=False, index=False)

    fusion_apfd = [apfd(idx_miss_list, fusion_rank_idx)]
    deepGini_apfd = [apfd(idx_miss_list, deepGini_rank_idx)]
    vanillasm_apfd = [apfd(idx_miss_list, vanillasm_rank_idx)]
    pcs_apfd = [apfd(idx_miss_list, pcs_rank_idx)]
    entropy_apfd = [apfd(idx_miss_list, entropy_rank_idx)]
    random_apfd = [apfd(idx_miss_list, random_rank_idx)]

    fusion_apfd.insert(0, subject_name + '_' + 'fusion')
    deepGini_apfd.insert(0, subject_name+'_'+'deepGini')
    vanillasm_apfd.insert(0, subject_name + '_' + 'vanillasm')
    pcs_apfd.insert(0, subject_name + '_' + 'pcs')
    entropy_apfd.insert(0, subject_name + '_' + 'entropy')
    random_apfd.insert(0, subject_name+'_'+'random')

    res_list = [fusion_apfd, deepGini_apfd, vanillasm_apfd, pcs_apfd, entropy_apfd, random_apfd]
    df = pd.DataFrame(columns=None, data=res_list)
    df.to_csv(path_result_apfd, mode='a', header=False, index=False)

    # Different model fusion methods
    y_pred_test_fusion_weight = y_pred_test_xgb*apfd(idx_miss_list_train, xgb_rank_idx_train) + y_pred_test_lr*apfd(idx_miss_list_train, lr_rank_idx_train) + \
                                y_pred_test_rf*apfd(idx_miss_list_train, rf_rank_idx_train) + y_pred_test_lgb*apfd(idx_miss_list_train, lgb_rank_idx_train)

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


