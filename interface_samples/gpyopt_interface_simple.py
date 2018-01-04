# -*- coding: utf-8 -*-

import GPyOpt

func = GPyOpt.objective_examples.experimentsNd.alpine1(input_dim=2) 

domains_def = [{"name": "x", "type": "continuous", "domain": (-2, 2)},
               {"name": "y", "type": "continuous", "domain": (0.0, 2.0)}]

def run(p_id, gpu_id, args):
    return func.f(args)

max_exp = 15
eps = 1e-2
