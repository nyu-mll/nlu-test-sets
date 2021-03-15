import argparse
import os
import pickle
import pandas as pd
import numpy as np
import torch

import pyro
import pyro.infer
import pyro.infer.mcmc
import pyro.distributions as dist

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def set_seeds(seed):
    np.random.seed(seed)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)

def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))

def main(args):
    # Set seed
    set_seeds(args.seed)

    # defining dataset sizes
    n_models, n_items = 18, 1000

    dist_mean = 0
    # define distribution parameters
    alpha_dist = {"mu": dist_mean, "std": args.alpha_std}
    theta_dist = {"mu": dist_mean, "std": args.item_param_std}

    positive_transform = lambda x: torch.log(1 + torch.exp(x))

    # Generate params
    betas = pyro.sample("b", dist.Normal(dist_mean * torch.ones(n_items, args.dimension), args.item_param_std))
    log_gamma = pyro.sample("log c", dist.Normal(dist_mean * torch.ones(n_items), args.item_param_std))
    gamma = sigmoid(log_gamma)

    alphas = pyro.sample("a",
                dist.LogNormal(
                    alpha_dist["mu"] * torch.ones(n_items, args.dimension),
                    alpha_dist["std"],
                ),
            )

    # Generate thetas
    thetas = pyro.sample(
                "theta",
                dist.Normal(
                    theta_dist["mu"] * torch.ones(n_models, args.dimension),
                    theta_dist["std"],
                ),
            )
    if args.dimension > 1:
        lik = dist.Bernoulli(
                    gamma[None, :]
                    + (1.0 - gamma[None, :])
                    * sigmoid(
                        1./np.sqrt(args.dimension) * torch.sum(alphas[None, :, :] * (thetas[:, None] - betas[None, :]).squeeze(), dim=-1)
                    )
                ).sample()
    else:
        lik = dist.Bernoulli(
            gamma[None, :]
            + (1.0 - gamma[None, :])
            * sigmoid(alphas[None, :] * (thetas[:, None] - betas[None, :]))
        ).sample()

    

    model_names = ["model_{}".format(i) for i in range(n_models)]
    items_names = [f"data_d{args.dimension}_mean{dist_mean}_a{args.alpha_std:.2f}_t{args.item_param_std:.2f}_{i}" for i in range(n_items)]

    df = pd.DataFrame(data=lik.numpy().astype(int),
                      index=model_names,
                      columns=items_names)
    df.index.names = ['userid']
    
    param_dict = {
		    "a": alphas.numpy(),
		    "b": betas.numpy(),
		    "g": gamma.numpy(),
                    "t": thetas.numpy()
		}
    with open(os.path.join(args.response_dir, f'params_sync_dim{args.dimension}_mean{dist_mean}_alpha-{args.discr}-{args.alpha_std:.2f}_theta-{args.ability}-{args.item_param_std:.2f}_irt_all_coded.p'), 'wb') as handle:
        pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    response_output=os.path.join(args.response_dir, f'sync_dim{args.dimension}_mean{dist_mean}_alpha-{args.discr}-{args.alpha_std:.2f}_theta-{args.ability}-{args.item_param_std:.2f}_irt_all_coded.csv')
    df.to_csv(response_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        "--response_dir", help="directory containing responses", required=True
    )
    parser.add_argument("--seed", default=42, help="random seed", type=int)

    # Distribution arguments
    distribution_choices = ["normal", "lognormal", "beta", "multivariate_normal"]
    transform_choices = ["identity", "positive"]

    parser.add_argument(
        "--diff",
        default="normal",
        help="difficulty (beta) distribution",
        choices=distribution_choices,
    )
    parser.add_argument(
        "--discr",
        default="lognormal",
        help="discrimination (alpha) distribution",
        choices=distribution_choices,
    )
    parser.add_argument(
        "--guess",
        default="normal",
        help="guessing (gamma) distribution",
        choices=distribution_choices,
    )
    parser.add_argument(
        "--ability",
        default="normal",
        help="ability (theta) distribution",
        choices=distribution_choices,
    )

    parser.add_argument(
        "--diff_transform",
        default="identity",
        help="difficulty (beta) transformation",
        choices=transform_choices,
    )
    parser.add_argument(
        "--discr_transform",
        default="identity",
        help="discrimination (alpha) transformation",
        choices=transform_choices,
    )
    parser.add_argument(
        "--guess_transform",
        default="identity",
        help="guessing (gamma) transformation",
        choices=transform_choices,
    )
    parser.add_argument(
        "--ability_transform",
        default="identity",
        help="ability (theta) transformation",
        choices=transform_choices,
    )

    parser.add_argument(
        "--alpha_std", default=1.0, type=float, help="standard deviation for alpha"
    )
    parser.add_argument(
        "--item_param_std",
        default=1.0,
        type=float,
        help="standard deviation for beta, log_gamma",
    )

    parser.add_argument("--dimension", default=3, help="dimension of IRT", type=int)
    parser.add_argument("--verbose", action="store_true", help="boolean for tracking")

    args = parser.parse_args()

    main(args)
