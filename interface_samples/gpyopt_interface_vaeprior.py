# -*- coding: utf-8 -*-

import os
import copy
import subprocess

import yaml
import numpy as np


domains_def = [{"name": "log_sgd_lr", "type": "continuous", "domain": (-10, -5.5)},
               {"name": "log_prior_s2", "type": "continuous", "domain": (-4.6, 0.0)}]
               #{"name": "sgd_start_epoch", "type": "discrete", "domain": (0, 25, 50)}]

default_cfg = {
    "model": "ImageGaussianPriorVAE",
    "inference_net_structure": [[64, [5, 5], [1, 1]], [128, [3, 3]], [128, [3, 3], [1, 1]], [256, [3, 3]], [256, [3, 3], [1,1]]],
    "softmax_topic_vector": False,
    "transfer_fct": "relu",
    "recon_transfer_fct": "sigmoid",

    "logvar_bias_init": 0,
    "logvar_weight_init": "xavier",

    "batch_norm_inference_net": False,
    "batch_norm_variational_param": False,
    "batch_norm_gen_net": False,

    "MC_samples": 1,

    "latent_dim": 200,

    "prior_mu": 0.0,
    "prior_lambda": 1.0,
    "prior_s1": 1.0,
    "prior_s2": 1.0
}
default_train_cfg = {
    "max_epochs": 900,
    "batch_size": 100,
    "optimizer": "AdamOptimizer",
    "optimizer_cfg": {
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
    },
    "kl_annealing": {
        "start": 1,
        "interval": 300,
        "step": 0.1,
        "max": 1,
    },
    "dropout_keep_prob": 1,
    "print_every": 1,
    "print_tensor_names": ["batch_rec_loss", "batch_kl_loss", "reg_loss", "bits_per_dim", "post_mu", "post_lambda", "post_gamma1", "post_gamma2"],
    # "save_tensor_every": 100,
    # "save_tensor_names": ["KL_loss_3d"],

    "test_batch_size": 100,
    "check_early_stop_every": 5,
    "stop_threshold": 50,
    "load_best_and_test": True,

    "use_structure": "conv",
    "conv_input_shape": [3, 32, 32],

    "image_loss_type": "discretized_logistic",

    "shuffle": True,

    # "snapshot_every": 150,

    "sgd_start_epoch": 0,
    "sgd_lr": 0.001
}

dataset = "cifar10_valid"
latent_dim = 1000

# make up some dirs
here = os.path.dirname(os.path.abspath(__file__))
bexp_cfg_dir = os.path.join(here, "bexp/configs")
bexp_res_dir = os.path.join(here, "bexp/results")
subprocess.check_call("mkdir -p {}".format(bexp_cfg_dir), shell=True)

run_f = os.path.join(here, "run_image_cfg_new.sh")

X_init=np.array([[-6.91, 0], [-9.21, 0], [-5.3, 0],
                 [-8.79, -0.22],
                 [-7.45, -2.35],
                 [-6.62, -1.72],
                 [-9.18, -1.53],
                 [-7.69, -0.47],
                 [-6.87, -0.33],
                 [-7.35, -1.96],
                 [-5.63, -1.50]])
Y_init=np.array([[11905.1],[11915.3],[np.inf],[11863.0],[11769.9], [11896.4],
                 [11705.5], [11707.1], [11881.9], [11812.0], [np.inf]])

def run(p_id, gpu_id, args):
    log_sgd_lr = args[0]
    log_prior_s2 = args[1]
    # sgd_start_epoch = args[2]
    sgd_lr = float(np.exp(log_sgd_lr))
    prior_s2 = float(np.exp(log_prior_s2))
    # sgd_start_epoch = int(sgd_start_epoch)
    cfg = copy.deepcopy(default_cfg)
    cfg["prior_s2"] = prior_s2
    train_cfg = copy.deepcopy(default_train_cfg)
    train_cfg["sgd_lr"] = sgd_lr
    # train_cfg["sgd_start_epoch"] = sgd_start_epoch
    # dump config to file
    # format_string = "{}_{}_{}".format(prior_s2, sgd_lr, sgd_start_epoch)
    format_string = "{:.3f}_{:.3f}".format(log_sgd_lr, log_prior_s2)
    cfg_dir = os.path.join(bexp_cfg_dir, format_string)
    os.mkdir(cfg_dir)
    with open(os.path.join(cfg_dir, "model.yaml"), "w") as f:
        yaml.dump(cfg, f)
    with open(os.path.join(cfg_dir, "train.yaml"), "w") as f:
        yaml.dump(train_cfg, f)
    res_dir = os.path.join(bexp_res_dir, format_string)
    d_res_dir = os.path.join(res_dir, dataset, format_string)
    subprocess.check_call("mkdir -p {}".format(d_res_dir), shell=True)
    error_log = os.path.join(d_res_dir, "error.log")
    
    run_cmd = "VAE_DATASET={dataset} CUDA_VISIBLE_DEVICES={gpu_id} VAE_SAVE_MODEL=1 bash {run_f} {cfg_dir} 1000dim {res_dir} {latent_dim} >/dev/null 2>{errorlog}".format(dataset=dataset, gpu_id=gpu_id, run_f=run_f, cfg_dir=cfg_dir, res_dir=res_dir, latent_dim=latent_dim, errorlog=error_log)
    try:
        subprocess.check_call(run_cmd, shell=True)
    except Exception as e:
        print(e)
        test_loss = np.inf
        return test_loss
    test_loss = float(subprocess.check_output("tail -n 1 {}".format(os.path.join(d_res_dir, "1000dim.log")) + " | awk '{print $NF}'", shell=True))
    return test_loss

eps = 1e-2
max_exp = 10
normalize_Y = False
