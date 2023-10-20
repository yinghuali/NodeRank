#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 100G

python mutation_node_feature.py --path_node_feature 'data/lastfm/x_np.pkl' --n_feature 10 --save_path_pkl 'data/lastfm/mutation_x_np_list.pkl'