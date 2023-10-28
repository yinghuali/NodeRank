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
    config.py: configuration script.
    feature_analysis.py: script of feature contribution.
    get_attack.py: script of attack dataset generation.
    get_mutation_models.py: script of model mutants generation.
    mutation_node_edge.py: script of graph structure mutants generation.
    mutation_node_feature.py: script of node feaure mutants generation.
    get_rank_idx.py: script for ranking.
    utils.py: tool script.
    main.py: main script.
    ----------------------
    mkdirFile.sh: directory preparation.
    structure_mutants.sh: graph structure mutants generation.
    node_feature_mutants.sh: node feature mutants generation.
    models_mutants.sh: graph models mutants generation.
    run_NodeRank.sh: run noderank to get all results.

    

## Usage
- Step1: Directory preparation:  
```sh mkdirFile.sh```

- Step2: Mutants Generation
    - Graph structure mutants generation  
    ```sh structure_mutants.sh```
    - Node feature mutants generation  
    ```sh node_feature_mutants.sh```
    - Graph models mutants generation  
    ```sh models_mutants.sh```
      
- Step3: Run NodeRank  
```sh run_NodeRank.sh```