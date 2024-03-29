
select_ratio_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]


hidden_channel_list = [5, 10, 15, 20] #

epochs_gcn = [30, 40, 50, 60, 70]
dic_mutation_gcn = {
    "normalize": [True, False],
    "bias": [True],
    "improved": [True, False],
    "cached": [True, False],
    "add_self_loops": [True, False]
}

epochs_gat = [30, 40, 50, 55, 60, 70, 80]
dic_mutation_gat = {
    "heads": [2, 4, 6, 8], #
    "concat": [True],
    "negative_slope": [0.1, 0.2, 0.3, 0.4], #
    "add_self_loops": [True, False],
    "bias": [True, False]
}

epochs_graphsage = [10, 20, 30, 45, 50, 70, 80]
dic_mutation_graphsage = {
    "normalize": [True, False],
    "bias": [True]
}


epochs_tagcn = [10, 20, 30, 40, 45, 45, 50, 60, 70, 80]
dic_mutation_tagcn = {
    "normalize": [True, False],
    "bias": [True]
}




