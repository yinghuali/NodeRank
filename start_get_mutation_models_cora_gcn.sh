#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 100G

python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_1/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_1/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_2/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_2/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_3/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_3/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_4/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_4/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_5/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_5/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_6/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_6/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_7/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_7/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_8/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_8/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_9/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_9/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_10/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_10/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_11/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_11/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_12/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_12/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_13/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_13/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_14/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_14/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_15/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_15/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_16/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_16/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_17/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_17/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_18/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_18/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_19/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_19/cora_gcn/cora_gcn_' --model_name 'gcn'
python get_mutation_models.py --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --path_save_model './new_mutation_models/repeat_20/cora_gcn/cora_gcn_' --path_save_config './new_mutation_models/repeat_20/cora_gcn/cora_gcn_' --model_name 'gcn'


