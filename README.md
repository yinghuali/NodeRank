# NodeRank
NodeRank is our proposed test prioritization approach specifically for GNNs.

## Environment setting
    PyTorch 1.11.0
    PyTorch Geometric 2.1.0
    XGBoost 1.4.2
    LighGBM 3.3.2
    scikit-learn 0.24.2

##  Repository catalogue
    data: processed and structured dataset.
    example_shell: scripts to obtain the experimental results of the paper. 
    models: scripts to get the GNN models.
    parameter_analysis: parameter analysis.
    results: experimental results of the paper.
    target_models: the evaluated GNN models.
    ----------------------
    config.py: configuration script.
    feature_analysis.py: script for ablation study 
    get_attack.py: script for generating adversarial dataset .
    get_mutation_models.py: script for generating GNN model mutants.
    mutation_node_edge.py: script for generating input mutants (graph structure mutation).
    mutation_node_feature.py: script for generating input mutants (node feature mutation).
    get_rank_idx.py: script for test input ranking.
    utils.py: tool script.
    main.py: main script.
    ----------------------
    mkdirFile.sh: directory preparation.
    structure_mutants.sh: graph structure mutant generation.
    node_feature_mutants.sh: node feature mutant generation.
    models_mutants.sh: GNN model mutant generation.
    run_NodeRank.sh: run NodeRank to get all results.

## How to replicate NodeRank
### Step1: Directory preparation:  
```sh mkdirFile.sh```

### Step2: Mutants Generation
- Graph structure mutant generation  
    ```sh structure_mutants.sh```
- Node feature mutant generation  
    ```sh node_feature_mutants.sh```
- GNN model mutant generation  
    ```sh models_mutants.sh```
### Step3: Run NodeRank  
```sh run_NodeRank.sh```