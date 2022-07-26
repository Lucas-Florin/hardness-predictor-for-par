import copy

from baseline_trainer import BaselineTrainer


def make_combinations(param_values_dict):
    param = sorted(list(param_values_dict.keys()))[0]
    values = param_values_dict[param]
    if len(param_values_dict) > 1:
        next_dict = dict(param_values_dict)
        del next_dict[param]
        combinations = make_combinations(next_dict)
    else: 
        combinations = None
    new_combinations = list()
    for v in values:
        comb = {param: v}
        if combinations is not None:
            new_combinations += [comb | old_comb for old_comb in combinations]
        else:
            new_combinations.append(comb)
    return new_combinations

def make_experiment_name(combination):
    params = sorted(combination.keys())
    s = 'grid-search'
    for p in params:
        s += '_'
        s += f'{p}_{combination[p]}'
    return s

def update_args(args, params):
    for p, v in params.items():
        setattr(args, p, v)
    return args

def run_grid_search(args, search_params):

    combinations = make_combinations(search_params)
    for comb in combinations: 
        comb.update({'experiment_name': make_experiment_name(comb)})
    args_combinations = [update_args(copy.copy(args), c) for c in combinations]


    for args in args_combinations:
        BaselineTrainer(args)

