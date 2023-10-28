# NodeRank
NodeRank is a test prioritization approach for GNNs.
## Main Requirements
    PyTorch 1.11.0
    PyTorch Geometric 2.1.0
    XGBoost 1.4.2
    LighGBM 3.3.2
    scikit-learn 0.24.2
## Catalogue of Repository
    data: processed and structured dataset.
    example_shell: scripts to obtain experimental results of paper. 
    models: scripts to get GNN models.
    results: experiment results of paper.
    target_models: GNN models.
    ----------------------
    config.py: configuration script
    feature_analysis.py: script of feature contribution.
    get_attack.py: script of attack dataset generation.
    get_mutation_models.py: script of model mutants generation.
    mutation_node_edge.py: script of graph structure mutants generation.
    mutation_node_feature.py: script of node feaure mutants generation.
    get_rank_idx.py: script for ranking.
    utils.py: tool script.
    main.py: main script.
    

## Usage
- Directory preparation:  
```sh mkdirFile.sh```

- Mutants Generation
    - Graph structure mutants generation  
    ```sh structure_mutants.sh```
    - Node feature mutants generation  
    ```sh node_feature_mutants.sh```
    - Graph models mutants generation  
    ```sh models_mutants.sh```
      
#### Run main
    python main.py --path_model_file './mutation_models/cora_gcn' --model_name 'gcn' --target_model_path './target_models/cora_gcn.pt' --path_x_np './data/cora/x_np.pkl' --path_edge_index './data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --subject_name 'cora_gcn' --path_mutation_edge_index_np_list './data/cora/mutation_edge_index_np_list.pkl' --path_mutation_x_np_list './data/cora/mutation_x_np_list.pkl'
