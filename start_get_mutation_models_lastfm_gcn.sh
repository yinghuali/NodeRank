#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 100G

python get_paramete_mutation_models.py --path_x_np './data/lastfm/x_np.pkl' --path_edge_index './data/lastfm/edge_index_np.pkl' --path_y './data/lastfm/y_np.pkl' --path_save_model './paramete_mutation_models/lastfm_gcn/lastfm_gcn_' --path_save_config './paramete_mutation_models/lastfm_gcn/lastfm_gcn_' --model_name 'gcn'


