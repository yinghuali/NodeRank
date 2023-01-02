import pandas as pd
import torch.nn.functional as F
import json
import torch
import datetime
import joblib
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

# python efficiency_analysis.py --path_model_file './mutation_models/cora_gcn' --model_name 'gcn' --target_model_path './target_models/cora_gcn.pt' --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --subject_name 'cora_gcn' --path_mutation_edge_index_np_list './data/cora/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/cora/mutation_x_np_list.pkl'
# python efficiency_analysis.py --path_model_file './mutation_models/citeseer_gcn' --model_name 'gcn' --target_model_path './target_models/citeseer_gcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index './data/citeseer/edge_index_np.pkl' --subject_name 'citeseer_gcn' --path_y './data/citeseer/y_np.pkl' --path_mutation_edge_index_np_list './data/citeseer/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/citeseer/mutation_x_np_list.pkl'
# python efficiency_analysis.py --path_model_file './mutation_models/lastfm_gcn' --model_name 'gcn' --target_model_path './target_models/lastfm_gcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index './data/lastfm/edge_index_np.pkl'  --subject_name 'lastfm_gcn' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
# python efficiency_analysis.py --path_model_file './mutation_models/pubmed_gcn' --model_name 'gcn' --target_model_path './target_models/pubmed_gcn.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index './data/pubmed/edge_index_np.pkl'  --subject_name 'pubmed_gcn' --path_y './data/pubmed/y_np.pkl' --path_mutation_edge_index_np_list './data/pubmed/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/pubmed/mutation_x_np_list.pkl'

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
starttime = datetime.datetime.now()

target_hidden_channel = 16
path_result_pfd = 'res/features/pfd' + '_' + subject_name + '.csv'
path_result_apfd = 'res/features/apfd' + '_' + subject_name + '.csv'


num_node_features, num_classes, x, edge_index, y, test_y, train_idx, test_idx = load_data(path_x_np, path_edge_index, path_y)

path_model_list = get_model_path(path_model_file)
path_model_list = sorted(path_model_list)
path_config_list = [i.replace('.pt', '.pkl') for i in path_model_list]
hidden_channel_list = [int(i.split('/')[-1].split('_')[2]) for i in path_config_list]
dic_list = [pickle.load(open(i, 'rb')) for i in path_config_list]

data_model_name = subject_name.split('_')[0]+'.model'


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()



model_list = []
for i in range(len(path_model_list)):
    try:
        tmp_model = load_model(model_name, path_model_list[i], hidden_channel_list[i], num_node_features,
                               num_classes, dic_list[i])
        model_list.append(tmp_model)
    except:
        print(dic_list[i])

target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes,
                                 target_model_path)

# mutant model feature
feature_model_np, label_model_np = get_mutation_model_features(num_node_features, target_hidden_channel,
                                                               num_classes, target_model_path, x, y, edge_index,
                                                               model_list, model_name)

# mutant node edge feature
mutation_edge_index_list = pickle.load(open(path_mutation_edge_index_np_list, 'rb'))
feature_node_edge_np, label_node_edge_np = get_mutation_nodel_edge_features(num_node_features,
                                                                            target_hidden_channel, num_classes,
                                                                            target_model_path, x, y, edge_index,
                                                                            mutation_edge_index_list, model_name)

# mutant node attribute feature
mutation_attribute_index_list = pickle.load(open(path_mutation_x_np_list, 'rb'))
feature_node_attribute_np, label_node_attribute_np = get_mutation_nodel_attribute_features(num_node_features,
                                                                                           target_hidden_channel,
                                                                                           num_classes,
                                                                                           target_model_path, x, y,
                                                                                           edge_index,
                                                                                           mutation_attribute_index_list,
                                                                                           model_name)

feature_np = np.concatenate((feature_model_np, feature_node_edge_np, feature_node_attribute_np), axis=1)
label_np = label_model_np

x_train = feature_np[train_idx]
y_train = label_np[train_idx]
x_test = feature_np[test_idx]
y_test = label_np[test_idx]

x_train = np.concatenate((feature_model_np, feature_node_edge_np, feature_node_attribute_np), axis=1)[train_idx]
x_test = np.concatenate((feature_model_np, feature_node_edge_np, feature_node_attribute_np), axis=1)[test_idx]

model = joblib.load('models/xgb_' + data_model_name)
y_pred_test_xgb = model.predict_proba(x_test)[:, 1]

model = joblib.load('models/lr_' + data_model_name)
y_pred_test_lr = model.predict_proba(x_test)[:, 1]

model = joblib.load('models/rf_' + data_model_name)
y_pred_test_rf = model.predict_proba(x_test)[:, 1]

model = joblib.load('models/lgb_' + data_model_name)
y_pred_test_lgb = model.predict_proba(x_test)[:, 1]

y_pred_test_fusion = y_pred_test_xgb + y_pred_test_lr + y_pred_test_rf + y_pred_test_lgb
fusion_rank_idx = y_pred_test_fusion.argsort()[::-1].copy()

# noderank time
noderank_time = datetime.datetime.now().second-starttime.second
write_result('noderank->'+str(noderank_time), 'time.txt')

margin_start = datetime.datetime.now()
num_node_features, num_classes, x, edge_index, y, test_y, train_idx, test_idx = load_data(path_x_np, path_edge_index, path_y)
target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes,
                                 target_model_path)
x_test_target_model_pre = target_model(x, edge_index).detach().numpy()[test_idx]
margin_rank_idx = Margin_rank_idx(x_test_target_model_pre)
margin_end = datetime.datetime.now()
margin_time = (margin_end.second - margin_start.second)
write_result('margin->' + str(margin_time), 'time.txt')

deepGini_start = datetime.datetime.now()
num_node_features, num_classes, x, edge_index, y, test_y, train_idx, test_idx = load_data(path_x_np, path_edge_index, path_y)
target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes,
                                 target_model_path)
x_test_target_model_pre = target_model(x, edge_index).detach().numpy()[test_idx]
deepGini_rank_idx = DeepGini_rank_idx(x_test_target_model_pre)
deepGini_end = datetime.datetime.now()
deepGini_time = (deepGini_end.second - deepGini_start.second)
write_result('deepGini->' + str(deepGini_time), 'time.txt')

leastConfidence_start = datetime.datetime.now()
num_node_features, num_classes, x, edge_index, y, test_y, train_idx, test_idx = load_data(path_x_np, path_edge_index, path_y)
target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes,
                                 target_model_path)
x_test_target_model_pre = target_model(x, edge_index).detach().numpy()[test_idx]
leastConfidence_rank_idx = MaximumProbability_rank_idx(x_test_target_model_pre)
leastConfidence_end = datetime.datetime.now()
leastConfidence_time = (leastConfidence_end.second - leastConfidence_start.second)
write_result('leastConfidence->' + str(leastConfidence_time), 'time.txt')





