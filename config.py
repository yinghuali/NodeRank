
select_ratio_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

hidden_channel_list = list(range(17, 20))


epochs_gcn = [60]
dic_mutation_gcn = {
    "normalize": [True, False],
    "bias": [True],
    "improved": [True, False],
    "cached": [True, False],
    "add_self_loops": [True, False]
}

epochs_gat = [40, 50, 55, 60]
dic_mutation_gat = {
    "heads": [4, 5],
    "concat": [True],
    "negative_slope": [0.1, 0.2],
    "add_self_loops": [True, False],
    "bias": [True, False]
}

epochs_graphsage = [10, 45, 50, 70]
dic_mutation_graphsage = {
    "normalize": [True, False],
    "bias": [True]
}


epochs_tagcn = [10, 45, 50, 70]
dic_mutation_tagcn = {
    "normalize": [True, False],
    "bias": [True]
}




