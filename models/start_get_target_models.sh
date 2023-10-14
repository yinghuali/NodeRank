#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 100G


python citeseer_gat_train.py --path_x_np '../data/citeseer/x_np.pkl'  --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y '../data/citeseer/y_np.pkl' --epochs 2 --save_model_name '../target_models/citeseer_gat.pt' --save_pre_name '../target_models/pre_np_citeseer_gat.pkl'
python citeseer_gcn_train.py --path_x_np '../data/citeseer/x_np.pkl'  --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y '../data/citeseer/y_np.pkl' --epochs 5 --save_model_name '../target_models/citeseer_gcn.pt' --save_pre_name '../target_models/pre_np_citeseer_gcn.pkl'
python citeseer_graphsage_train.py --path_x_np '../data/citeseer/x_np.pkl'  --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y '../data/citeseer/y_np.pkl' --epochs 5 --save_model_name '../target_models/citeseer_graphsage.pt' --save_pre_name '../target_models/pre_np_citeseer_graphsage.pkl'
python citeseer_tagcn_train.py --path_x_np '../data/citeseer/x_np.pkl'  --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y '../data/citeseer/y_np.pkl' --epochs 3 --save_model_name '../target_models/citeseer_tagcn.pt' --save_pre_name '../target_models/pre_np_citeseer_tagcn.pkl'

python cora_gat_train.py --path_x_np '../data/cora/x_np.pkl'  --path_edge_index '../data/cora/edge_index_np.pkl' --path_y '../data/cora/y_np.pkl' --epochs 4 --save_model_name '../target_models/cora_gat.pt' --save_pre_name '../target_models/pre_np_cora_gat.pkl'
python cora_gcn_train.py --path_x_np '../data/cora/x_np.pkl'  --path_edge_index '../data/cora/edge_index_np.pkl' --path_y '../data/cora/y_np.pkl' --epochs 10 --save_model_name '../target_models/cora_gcn.pt' --save_pre_name '../target_models/pre_np_cora_gcn.pkl'
python cora_graphsage_train.py --path_x_np '../data/cora/x_np.pkl'  --path_edge_index '../data/cora/edge_index_np.pkl' --path_y '../data/cora/y_np.pkl' --epochs 7 --save_model_name '../target_models/cora_graphsage.pt' --save_pre_name '../target_models/pre_np_cora_graphsage.pkl'
python cora_tagcn_train.py --path_x_np '../data/cora/x_np.pkl'  --path_edge_index '../data/cora/edge_index_np.pkl' --path_y '../data/cora/y_np.pkl' --epochs 5 --save_model_name '../target_models/cora_tagcn.pt' --save_pre_name '../target_models/pre_np_cora_tagcn.pkl'

python lastfm_gat_train.py --path_x_np '../data/lastfm/x_np.pkl'  --path_edge_index '../data/lastfm/edge_index_np.pkl' --path_y '../data/lastfm/y_np.pkl' --epochs 10 --save_model_name '../target_models/lastfm_gat.pt' --save_pre_name '../target_models/pre_np_lastfm_gat.pkl'
python lastfm_gcn_train.py --path_x_np '../data/lastfm/x_np.pkl'  --path_edge_index '../data/lastfm/edge_index_np.pkl' --path_y '../data/lastfm/y_np.pkl' --epochs 15 --save_model_name '../target_models/lastfm_gcn.pt' --save_pre_name '../target_models/pre_np_lastfm_gcn.pkl'
python lastfm_graphsage_train.py --path_x_np '../data/lastfm/x_np.pkl'  --path_edge_index '../data/lastfm/edge_index_np.pkl' --path_y '../data/lastfm/y_np.pkl' --epochs 28 --save_model_name '../target_models/lastfm_graphsage.pt' --save_pre_name '../target_models/pre_np_lastfm_graphsage.pkl'
python lastfm_tagcn_train.py --path_x_np '../data/lastfm/x_np.pkl'  --path_edge_index '../data/lastfm/edge_index_np.pkl' --path_y '../data/lastfm/y_np.pkl' --epochs 37 --save_model_name '../target_models/lastfm_tagcn.pt' --save_pre_name '../target_models/pre_np_lastfm_tagcn.pkl'

python pubmed_gat_train.py --path_x_np '../data/pubmed/x_np.pkl'  --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y '../data/pubmed/y_np.pkl' --epochs 10 --save_model_name '../target_models/pubmed_gat.pt' --save_pre_name '../target_models/pre_np_pubmed_gat.pkl'
python pubmed_gcn_train.py --path_x_np '../data/pubmed/x_np.pkl'  --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y '../data/pubmed/y_np.pkl' --epochs 20 --save_model_name '../target_models/pubmed_gcn.pt' --save_pre_name '../target_models/pre_np_pubmed_gcn.pkl'
python pubmed_graphsage_train.py --path_x_np '../data/pubmed/x_np.pkl'  --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y '../data/pubmed/y_np.pkl' --epochs 15 --save_model_name '../target_models/pubmed_graphsage.pt' --save_pre_name '../target_models/pre_np_pubmed_graphsage.pkl'
python pubmed_tagcn_train.py --path_x_np '../data/pubmed/x_np.pkl'  --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y '../data/pubmed/y_np.pkl' --epochs 10 --save_model_name '../target_models/pubmed_tagcn.pt' --save_pre_name '../target_models/pre_np_pubmed_tagcn.pkl'