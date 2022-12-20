#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 100G

python main.py --path_model_file './mutation_models/lastfm_gcn' --model_name 'gcn' --target_model_path './target_models/lastfm_gcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index './data/lastfm/edge_index_np.pkl'  --subject_name 'lastfm_gcn' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_gcn' --model_name 'gcn' --target_model_path './target_models/lastfm_gcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_dice.pkl' --subject_name 'lastfm_gcn_dice' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_gcn' --model_name 'gcn' --target_model_path './target_models/lastfm_gcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_nodeembeddingattack_add.pkl' --subject_name 'lastfm_gcn_nodeembeddingattack_add' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_gcn' --model_name 'gcn' --target_model_path './target_models/lastfm_gcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_nodeembeddingattack_remove.pkl' --subject_name 'lastfm_gcn_nodeembeddingattack_remove' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_gcn' --model_name 'gcn' --target_model_path './target_models/lastfm_gcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_pgdattack.pkl' --subject_name 'lastfm_gcn_pgdattack' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_gcn' --model_name 'gcn' --target_model_path './target_models/lastfm_gcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_randomattack_add.pkl' --subject_name 'lastfm_gcn_randomattack_add' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_gcn' --model_name 'gcn' --target_model_path './target_models/lastfm_gcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_randomattack_flip.pkl' --subject_name 'lastfm_gcn_randomattack_flip' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_gcn' --model_name 'gcn' --target_model_path './target_models/lastfm_gcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_randomattack_remove.pkl' --subject_name 'lastfm_gcn_randomattack_remove' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'


python main.py --path_model_file './mutation_models/lastfm_gat' --model_name 'gat' --target_model_path './target_models/lastfm_gat.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index './data/lastfm/edge_index_np.pkl'  --subject_name 'lastfm_gat' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_gat' --model_name 'gat' --target_model_path './target_models/lastfm_gat.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_dice.pkl' --subject_name 'lastfm_gat_dice' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_gat' --model_name 'gat' --target_model_path './target_models/lastfm_gat.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_nodeembeddingattack_add.pkl' --subject_name 'lastfm_gat_nodeembeddingattack_add' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_gat' --model_name 'gat' --target_model_path './target_models/lastfm_gat.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_nodeembeddingattack_remove.pkl' --subject_name 'lastfm_gat_nodeembeddingattack_remove' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_gat' --model_name 'gat' --target_model_path './target_models/lastfm_gat.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_pgdattack.pkl' --subject_name 'lastfm_gat_pgdattack' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_gat' --model_name 'gat' --target_model_path './target_models/lastfm_gat.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_randomattack_add.pkl' --subject_name 'lastfm_gat_randomattack_add' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_gat' --model_name 'gat' --target_model_path './target_models/lastfm_gat.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_randomattack_flip.pkl' --subject_name 'lastfm_gat_randomattack_flip' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_gat' --model_name 'gat' --target_model_path './target_models/lastfm_gat.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_randomattack_remove.pkl' --subject_name 'lastfm_gat_randomattack_remove' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'

python main.py --path_model_file './mutation_models/lastfm_tagcn' --model_name 'tagcn' --target_model_path './target_models/lastfm_tagcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index './data/lastfm/edge_index_np.pkl'  --subject_name 'lastfm_tagcn' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_tagcn' --model_name 'tagcn' --target_model_path './target_models/lastfm_tagcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_dice.pkl' --subject_name 'lastfm_tagcn_dice' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_tagcn' --model_name 'tagcn' --target_model_path './target_models/lastfm_tagcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_nodeembeddingattack_add.pkl' --subject_name 'lastfm_tagcn_nodeembeddingattack_add' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_tagcn' --model_name 'tagcn' --target_model_path './target_models/lastfm_tagcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_nodeembeddingattack_remove.pkl' --subject_name 'lastfm_tagcn_nodeembeddingattack_remove' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_tagcn' --model_name 'tagcn' --target_model_path './target_models/lastfm_tagcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_pgdattack.pkl' --subject_name 'lastfm_tagcn_pgdattack' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_tagcn' --model_name 'tagcn' --target_model_path './target_models/lastfm_tagcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_randomattack_add.pkl' --subject_name 'lastfm_tagcn_randomattack_add' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_tagcn' --model_name 'tagcn' --target_model_path './target_models/lastfm_tagcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_randomattack_flip.pkl' --subject_name 'lastfm_tagcn_randomattack_flip' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_tagcn' --model_name 'tagcn' --target_model_path './target_models/lastfm_tagcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_randomattack_remove.pkl' --subject_name 'lastfm_tagcn_randomattack_remove' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'

python main.py --path_model_file './mutation_models/lastfm_graphsage' --model_name 'graphsage' --target_model_path './target_models/lastfm_graphsage.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index './data/lastfm/edge_index_np.pkl'  --subject_name 'lastfm_graphsage' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_graphsage' --model_name 'graphsage' --target_model_path './target_models/lastfm_graphsage.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_dice.pkl' --subject_name 'lastfm_graphsage_dice' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_graphsage' --model_name 'graphsage' --target_model_path './target_models/lastfm_graphsage.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_nodeembeddingattack_add.pkl' --subject_name 'lastfm_graphsage_nodeembeddingattack_add' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_graphsage' --model_name 'graphsage' --target_model_path './target_models/lastfm_graphsage.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_nodeembeddingattack_remove.pkl' --subject_name 'lastfm_graphsage_nodeembeddingattack_remove' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_graphsage' --model_name 'graphsage' --target_model_path './target_models/lastfm_graphsage.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_pgdattack.pkl' --subject_name 'lastfm_graphsage_pgdattack' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_graphsage' --model_name 'graphsage' --target_model_path './target_models/lastfm_graphsage.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_randomattack_add.pkl' --subject_name 'lastfm_graphsage_randomattack_add' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_graphsage' --model_name 'graphsage' --target_model_path './target_models/lastfm_graphsage.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_randomattack_flip.pkl' --subject_name 'lastfm_graphsage_randomattack_flip' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'
python main.py --path_model_file './mutation_models/lastfm_graphsage' --model_name 'graphsage' --target_model_path './target_models/lastfm_graphsage.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index  './data/attack_data/lastfm/lastfm_randomattack_remove.pkl' --subject_name 'lastfm_graphsage_randomattack_remove' --path_y './data/lastfm/y_np.pkl' --path_mutation_edge_index_np_list './data/lastfm/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/lastfm/mutation_x_np_list.pkl'







