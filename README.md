# Source Code of NodeRank
## Main Requirements
    PyTorch 1.11.0
    PyTorch Geometric 2.1.0
    XGBoost 1.4.2
    LighGBM 3.3.2
    scikit-learn 0.24.2

#### Test attack data generation
    python get_attack.py --path_x_np './data/pubmed/x_np.pkl' --path_edge_index './data/pubmed/edge_index_np.pkl' --path_y './data/pubmed/y_np.pkl' --save_edge_index './test/attack_data/pubmed/pubmed'

#### Test mutation models generation
    python get_mutation_models.py --path_x_np './data/lastfm/x_np.pkl' --path_edge_index './data/lastfm/edge_index_np.pkl' --path_y './data/lastfm/y_np.pkl' --path_save_model './test/lastfm_tagcn/lastfm_tagcn_' --path_save_config './test/lastfm_tagcn/lastfm_tagcn_' --model_name 'tagcn'

#### Test mutation node edge generation
    python mutation_node_edge.py --path_edge_index_np 'data/pubmed/edge_index_np.pkl' --n_edges 50 --save_path_pkl 'data/pubmed/mutation_edge_index_np_list.pkl'

#### Test mutation node feature generation
    python mutation_node_feature.py --path_node_feature 'data/pubmed/x_np.pkl' --n_feature 50 --save_path_pkl 'data/pubmed/mutation_x_np_list.pkl'

#### Test main
    python main.py --path_model_file './mutation_models/cora_gcn' --model_name 'gcn' --target_model_path './target_models/cora_gcn.pt' --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --subject_name 'cora_gcn' --path_mutation_edge_index_np_list './data/cora/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/cora/mutation_x_np_list.pkl'
