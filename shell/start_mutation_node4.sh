#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 100G

python mutation_node_edge.py --path_edge_index_np 'data/pubmed/edge_index_np.pkl' --n_edges 50 --save_path_pkl 'data/pubmed/mutation_edge_index_np_list.pkl'