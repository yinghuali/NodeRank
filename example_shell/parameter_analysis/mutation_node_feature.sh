#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 100G


python parameter_mutation_node_feature.py --path_node_feature 'data/cora/x_np.pkl' --n_feature 50 --save_path_pkl 'data/cora/mutation_x_np_list.pkl'
python parameter_mutation_node_feature.py --path_node_feature 'data/citeseer/x_np.pkl' --n_feature 50 --save_path_pkl 'data/citeseer/mutation_x_np_list.pkl'
python parameter_mutation_node_feature.py --path_node_feature 'data/pubmed/x_np.pkl' --n_feature 50 --save_path_pkl 'data/pubmed/mutation_x_np_list.pkl'
python parameter_mutation_node_feature.py --path_node_feature 'data/lastfm/x_np.pkl' --n_feature 10 --save_path_pkl 'data/lastfm/mutation_x_np_list.pkl'