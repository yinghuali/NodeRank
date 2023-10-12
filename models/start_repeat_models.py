import os

for i in range(1, 21):
    cmd = "python citeseer_gat_train.py --path_x_np '../data/citeseer/x_np.pkl'  --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y '../data/citeseer/y_np.pkl' --epochs 5 --save_model_name '../repeat_target_models/repeat_{}/citeseer_gat.pt' --save_pre_name '../repeat_target_models/repeat_{}/pre_np_citeseer_gat.pkl'".format(str(i), str(i))
    os.system(cmd)


for i in range(1, 21):
    cmd = "python citeseer_gcn_train.py --path_x_np '../data/citeseer/x_np.pkl'  --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y '../data/citeseer/y_np.pkl' --epochs 5 --save_model_name '../repeat_target_models/repeat_{}/citeseer_gcn.pt' --save_pre_name '../repeat_target_models/repeat_{}/pre_np_citeseer_gcn.pkl'".format(str(i), str(i))
    os.system(cmd)


for i in range(1, 21):
    cmd = "python citeseer_graphsage_train.py --path_x_np '../data/citeseer/x_np.pkl'  --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y '../data/citeseer/y_np.pkl' --epochs 5 --save_model_name '../repeat_target_models/repeat_{}/citeseer_graphsage.pt' --save_pre_name '../repeat_target_models/repeat_{}/pre_np_citeseer_graphsage.pkl'".format(str(i), str(i))
    os.system(cmd)


for i in range(1, 21):
    cmd = "python citeseer_tagcn_train.py --path_x_np '../data/citeseer/x_np.pkl'  --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y '../data/citeseer/y_np.pkl' --epochs 5 --save_model_name '../repeat_target_models/repeat_{}/citeseer_tagcn.pt' --save_pre_name '../repeat_target_models/repeat_{}/pre_np_citeseer_tagcn.pkl'".format(str(i), str(i))
    os.system(cmd)


for i in range(1, 21):
    cmd = "python cora_gat_train.py --path_x_np '../data/cora/x_np.pkl'  --path_edge_index '../data/cora/edge_index_np.pkl' --path_y '../data/cora/y_np.pkl' --epochs 5 --save_model_name '../repeat_target_models/repeat_{}/cora_gat.pt' --save_pre_name '../repeat_target_models/repeat_{}/pre_np_cora_gat.pkl'".format(str(i), str(i))
    os.system(cmd)


for i in range(1, 21):
    cmd = "python cora_gcn_train.py --path_x_np '../data/cora/x_np.pkl'  --path_edge_index '../data/cora/edge_index_np.pkl' --path_y '../data/cora/y_np.pkl' --epochs 10 --save_model_name '../repeat_target_models/repeat_{}/cora_gcn.pt' --save_pre_name '../repeat_target_models/repeat_{}/pre_np_cora_gcn.pkl'".format(str(i), str(i))
    os.system(cmd)


for i in range(1, 21):
    cmd = "python cora_graphsage_train.py --path_x_np '../data/cora/x_np.pkl'  --path_edge_index '../data/cora/edge_index_np.pkl' --path_y '../data/cora/y_np.pkl' --epochs 10 --save_model_name '../repeat_target_models/repeat_{}/cora_graphsage.pt' --save_pre_name '../repeat_target_models/repeat_{}/pre_np_cora_graphsage.pkl'".format(str(i), str(i))
    os.system(cmd)


for i in range(1, 21):
    cmd = "python cora_tagcn_train.py --path_x_np '../data/cora/x_np.pkl'  --path_edge_index '../data/cora/edge_index_np.pkl' --path_y '../data/cora/y_np.pkl' --epochs 5 --save_model_name '../repeat_target_models/repeat_{}/cora_tagcn.pt' --save_pre_name '../repeat_target_models/repeat_{}/pre_np_cora_tagcn.pkl'".format(str(i), str(i))
    os.system(cmd)


for i in range(1, 21):
    cmd = "python lastfm_gat_train.py --path_x_np '../data/lastfm/x_np.pkl'  --path_edge_index '../data/lastfm/edge_index_np.pkl' --path_y '../data/lastfm/y_np.pkl' --epochs 10 --save_model_name '../repeat_target_models/repeat_{}/lastfm_gat.pt' --save_pre_name '../repeat_target_models/repeat_{}/pre_np_lastfm_gat.pkl'".format(str(i), str(i))
    os.system(cmd)

for i in range(1, 21):
    cmd = "python lastfm_gcn_train.py --path_x_np '../data/lastfm/x_np.pkl'  --path_edge_index '../data/lastfm/edge_index_np.pkl' --path_y '../data/lastfm/y_np.pkl' --epochs 20 --save_model_name '../repeat_target_models/repeat_{}/lastfm_gcn.pt' --save_pre_name '../repeat_target_models/repeat_{}/pre_np_lastfm_gcn.pkl'".format(str(i), str(i))
    os.system(cmd)


for i in range(1, 21):
    cmd = "python lastfm_graphsage_train.py --path_x_np '../data/lastfm/x_np.pkl'  --path_edge_index '../data/lastfm/edge_index_np.pkl' --path_y '../data/lastfm/y_np.pkl' --epochs 30 --save_model_name '../repeat_target_models/repeat_{}/lastfm_graphsage.pt' --save_pre_name '../repeat_target_models/repeat_{}/pre_np_lastfm_graphsage.pkl'".format(str(i), str(i))
    os.system(cmd)


for i in range(1, 21):
    cmd = "python lastfm_tagcn_train.py --path_x_np '../data/lastfm/x_np.pkl'  --path_edge_index '../data/lastfm/edge_index_np.pkl' --path_y '../data/lastfm/y_np.pkl' --epochs 40 --save_model_name '../repeat_target_models/repeat_{}/lastfm_tagcn.pt' --save_pre_name '../repeat_target_models/repeat_{}/pre_np_lastfm_tagcn.pkl'".format(str(i), str(i))
    os.system(cmd)


for i in range(1, 21):
    cmd = "python pubmed_gat_train.py --path_x_np '../data/pubmed/x_np.pkl'  --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y '../data/pubmed/y_np.pkl' --epochs 10 --save_model_name '../repeat_target_models/repeat_{}/pubmed_gat.pt' --save_pre_name '../repeat_target_models/repeat_{}/pre_np_pubmed_gat.pkl'".format(str(i), str(i))
    os.system(cmd)

for i in range(1, 21):
    cmd = "python pubmed_gcn_train.py --path_x_np '../data/pubmed/x_np.pkl'  --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y '../data/pubmed/y_np.pkl' --epochs 20 --save_model_name '../repeat_target_models/repeat_{}/pubmed_gcn.pt' --save_pre_name '../repeat_target_models/repeat_{}/pre_np_pubmed_gcn.pkl'".format(str(i), str(i))
    os.system(cmd)


for i in range(1, 21):
    cmd = "python pubmed_graphsage_train.py --path_x_np '../data/pubmed/x_np.pkl'  --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y '../data/pubmed/y_np.pkl' --epochs 15 --save_model_name '../repeat_target_models/repeat_{}/pubmed_graphsage.pt' --save_pre_name '../repeat_target_models/repeat_{}/pre_np_pubmed_graphsage.pkl'".format(str(i), str(i))
    os.system(cmd)


for i in range(1, 21):
    cmd = "python pubmed_tagcn_train.py --path_x_np '../data/pubmed/x_np.pkl'  --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y '../data/pubmed/y_np.pkl' --epochs 10 --save_model_name '../repeat_target_models/repeat_{}/pubmed_tagcn.pt' --save_pre_name '../repeat_target_models/repeat_{}/pre_np_pubmed_tagcn.pkl'".format(str(i), str(i))
    os.system(cmd)