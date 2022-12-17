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

path_model_file = './mutation_models/cora_gcn'
model_name = 'gcn'
target_model_path = './target_models/cora_gcn.pt'
path_x_np = './data/cora/x_np.pkl'
path_edge_index = './data/cora/edge_index_np.pkl'
path_y = './data/cora/y_np.pkl'
subject_name = 'cora_gcn'
path_mutation_edge_index_np_list = './data/cora/mutation_edge_index_np_list.pkl'
path_mutation_x_np_list = './data/cora/mutation_x_np_list.pkl'


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
    y_pred_train_xgb = model.predict_proba(x_train)[:, 1]
    y_pred_test_xgb = model.predict_proba(x_test)[:, 1]
    xgb_rank_idx = y_pred_test_xgb.argsort()[::-1].copy()

    y_pre = model.predict(x_train)
    print(y_pre.shape)
    print(y_pre[:10])

    # LR
    model = LogisticRegression(solver='liblinear')
    model.fit(x_train, y_train)
    y_pred_train_lr = model.predict_proba(x_train)[:, 1]
    y_pred_test_lr = model.predict_proba(x_test)[:, 1]
    lr_rank_idx = y_pred_test_lr.argsort()[::-1].copy()

    y_pre = model.predict(x_train)
    print(y_pre.shape)
    print(y_pre[:10])


    #
    # # RF
    # model = RandomForestClassifier()
    # model.fit(x_train, y_train)
    # y_pred_train_rf = model.predict_proba(x_train)[:, 1]
    # y_pred_test_rf = model.predict_proba(x_test)[:, 1]
    # rf_rank_idx = y_pred_test_rf.argsort()[::-1].copy()
    #
    # # LGBM
    # model = LGBMClassifier()
    # model.fit(x_train, y_train)
    # y_pred_train_lgb = model.predict_proba(x_train)[:, 1]
    # y_pred_test_lgb = model.predict_proba(x_test)[:, 1]
    # lgb_rank_idx = y_pred_test_lgb.argsort()[::-1].copy()





if __name__ == '__main__':
    main()



