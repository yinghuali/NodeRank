#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH --output=/dev/null
#SBATCH -p batch

python mutation_node_edge.py --path_edge_index_np 'data/citeseer/edge_index_np.pkl' --n_edges 480 --save_path_pkl 'data/citeseer/mutation_edge_index_np_list.pkl'