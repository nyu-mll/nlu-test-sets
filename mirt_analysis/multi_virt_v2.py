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

from pyro.infer import SVI, Trace_ELBO
from tqdm.auto import tqdm
from weighted_ELBO import Weighted_Trace_ELBO, WeightedSVI

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def irt_model(
    obs,
    alpha_dist,
    theta_dist,
    alpha_transform=lambda x: x,
    theta_transform=lambda x: x,
    item_params_std=1.0,
    dimension = 1,
    num_factors=3
):
    '''
    3 parameter IRT model used for stochastic variational inference. The model is defined by the
    distributions of the 3 item parameters (discrimination [a], difficulty [b], and
    guessing [g]) and ability parameter [t].

    The difficuly and log-guessing parameters follow Gaussian distributions with mean 0 and
    standard deviation `item_params_std`. Discrimination and ability parameters follow distributions
    defined by `alpha_dist` and `theta_dist`, respectively, followed by a transformation defined by
    `alpha_transform` and `theta_transform`, respectively.

    Args:
        obs:             Numpy array of item responses.
        alpha_dist:      {`name`: distribution_name, `param`: distribution_dict}
                         Dictionary for the distribution type for the discrimation parameter [alpha].
                         distribution_dict is a dictionary of distribution parameters necessary for the
                         parameteric distribution defined by distribution_name.
        theta_dist:      {`name`: distribution_name, `param`: distribution_dict}
                         Dictionary for the distribution type for the ability parameter [theta].
                         distribution_dict is a dictionary of distribution parameters necessary for the
                         parameteric distribution defined by distribution_name.
        alpha_transform: Transformation applied to the discrimination parameter [alpha].
        theta_transform: Transformation aplied to the ability parameter [theta].
        item_params_std: Standard deviation for difficulty and guessing parameters.

    Returns:
        lik:             the log-likelihood of item responses from obs given the
                         estimated parameters.
    '''
    n_models, n_items = obs.shape[0], obs.shape[1]

    betas = pyro.sample(
        "b",
        dist.MultivariateNormal(
            torch.zeros(n_items, dimension),
            scale_tril=item_params_std*torch.stack([torch.eye(dimension)]*n_items))
    )
    if num_factors >= 3:
        log_gamma = pyro.sample("log c", dist.Normal(torch.zeros(n_items), item_params_std))
        gamma = sigmoid(log_gamma)
    else:
        gamma = torch.ones(n_items) * 0.5

    if num_factors < 2:
        alphas = torch.ones(n_items, dimension)
    elif alpha_dist["name"] == "normal" or alpha_dist["name"] == "lognormal":
        alphas = pyro.sample(
            "a",
            dist.MultivariateNormal(
                alpha_dist["param"]["mu"] * torch.ones(n_items, dimension),
                scale_tril=alpha_dist["param"]["std"] * torch.stack([torch.eye(dimension)]*n_items),
            ),
        )
    # elif alpha_dist["name"] == "lognormal":
    #     alphas = pyro.sample(
    #         "a",
    #         torch.exp(dist.MultivariateNormal(
    #             alpha_dist["param"]["mu"] * torch.ones(n_items, dimension),
    #             scale_tril=alpha_dist["param"]["std"] * torch.stack([torch.eye(dimension)]*n_items),
    #         )),
    #     )
    elif alpha_dist["name"] == "beta":
        alphas = pyro.sample(
            "a",
            dist.Beta(
                alpha_dist["param"]["alpha"] * torch.ones(n_items, dimension),
                alpha_dist["param"]["beta"] * torch.ones(n_items, dimension),
            ),
        )
    else:
        raise TypeError(f"Alpha distribution {alpha_dist['name']} not supported.")

    if theta_dist["name"] == "normal" or theta_dist["name"] == "lognormal":
        thetas = pyro.sample(
            "theta",
            dist.MultivariateNormal(
                theta_dist["param"]["mu"] * torch.ones(n_models, dimension),
                scale_tril=theta_dist["param"]["std"] * torch.stack([torch.eye(dimension)]*n_models),
            ),
        )
    # elif theta_dist["name"] == "lognormal":
    #     thetas = pyro.sample(
    #         "theta",
    #         torch.exp(dist.MultivariateNormal(
    #             theta_dist["param"]["mu"] * torch.ones(n_models, dimension),
    #             scale_tril=theta_dist["param"]["std"] * torch.stack([torch.eye(dimension)] * n_models),
    #         )),
    #     )
    elif theta_dist["name"] == "beta":
        thetas = pyro.sample(
            "theta",
            dist.Beta(
                theta_dist["param"]["alpha"] * torch.ones(n_models, dimension),
                theta_dist["param"]["beta"] * torch.ones(n_models, dimension),
            ),
        )
    else:
        raise TypeError(f"theta distribution {theta_dist['name']} not supported.")


    if num_factors > 2:
        alphas = alpha_transform(alphas)
    thetas = theta_transform(thetas)

    if alpha_dist["name"] == "lognormal":
        alphas = torch.exp(alphas)

    if theta_dist["name"] == "lognormal":
        thetas = torch.exp(thetas)

    ################################ DEBUG ################################################
    # assert False, f"debug {(thetas[:, None, :] - betas[None, :, :]).shape} | {alphas[None, :, :].shape}"

    if dimension > 1:
        prob = (gamma[None, :]
            + (1.0 - gamma[None, :])
            * sigmoid(
                torch.sum(alphas[None, :, :] * (thetas[:, None] - betas[None, :]).squeeze(), dim=-1)
            ))
    else:
        betas = betas.squeeze()
        gamma = gamma.squeeze()
        alphas = alphas.squeeze()
        thetas = thetas.squeeze()
        prob = gamma[None, :] + (1.0 - gamma[None, :]) * sigmoid(alphas[None, :] * (thetas[:, None] - betas[None, :]))

    lik = pyro.sample(
        "likelihood",
        dist.Bernoulli(prob),
        obs=obs,
    )

    return lik


def vi_posterior(obs, alpha_dist, theta_dist, dimension=1, num_factors=3):
    '''
    3 parameter IRT guide used for stochastic variational inference in Pyro.

    Difficulty [b] and log-guessing [log c] follow Gaussian distributions. Discrimination [a] and
    ability [theta] follow distributions defined by `alpha_dist` and `theta_dist`, respectively.

    Args:
        obs:             Numpy array of item responses.
        alpha_dist:      {`name`: distribution_name, `param`: distribution_dict}
                         Dictionary for the distribution type for the discrimation parameter [alpha].
                         distribution_dict is a dictionary of distribution parameters necessary for the
                         parameteric distribution defined by distribution_name.
        theta_dist:      {`name`: distribution_name, `param`: distribution_dict}
                         Dictionary for the distribution type for the ability parameter [theta].
                         distribution_dict is a dictionary of distribution parameters necessary for the
                         parameteric distribution defined by distribution_name.
    '''
    n_models, n_items = obs.shape[0], obs.shape[1]

    # logstd is now lower_triangular matrix of cholesky decomposition

    # log_cov_template = torch.tensor(np.fill_diagonal(float("-inf")*np.ones((dimension, dimension)), 1))
    log_cov_template = torch.eye(dimension)

    pyro.sample(
        "b",
        dist.MultivariateNormal(
            pyro.param("b mu", torch.zeros(n_items, dimension)),
            scale_tril=torch.tril(pyro.param("b logstd", torch.stack([log_cov_template] * n_items))),
        ),
    )

    if num_factors >= 3:
        pyro.sample(
            "log c",
            dist.Normal(
                pyro.param("g mu", torch.zeros(n_items)),
                torch.exp(pyro.param("g logstd", torch.zeros(n_items))),
            ),
        )

    if num_factors >= 2 and (alpha_dist["name"] == "normal" or alpha_dist["name"] == "lognormal"):
        pyro.sample(
            "a",
            dist.MultivariateNormal(
                pyro.param("a mu", alpha_dist["param"]["mu"] * torch.ones(n_items, dimension)),
                scale_tril=torch.tril(
                    pyro.param(
                        "a logstd",
                        torch.tensor(alpha_dist["param"]["std"])
                        * torch.stack([log_cov_template] * n_items),
                    )
                ),
            ),
        )
    # elif alpha_dist["name"] == "lognormal":
    #     pyro.sample(
    #         "a",
    #         torch.exp(dist.MultivariateNormal(
    #             pyro.param("a mu", alpha_dist["param"]["mu"] * torch.ones(n_items, dimension)),
    #             scale_tril=torch.exp(
    #                 pyro.param(
    #                     "a logstd",
    #                     torch.log(torch.tensor(alpha_dist["param"]["std"]))
    #                     * torch.stack([log_cov_template] * n_items),
    #                 )
    #             ),
    #         )),
    #     )
    elif num_factors >= 2 and (alpha_dist["name"] == "beta"):
        pyro.sample(
            "a",
            dist.Beta(
                pyro.param(
                    "a alpha", alpha_dist["param"]["alpha"] * torch.ones(n_items, dimension)
                ),
                pyro.param("a beta", alpha_dist["param"]["beta"] * torch.ones(n_items, dimension)),
            ),
        )
    elif num_factors >= 2:
        raise TypeError(f"Alpha distribution {alpha_dist['name']} not supported.")

    if theta_dist["name"] == "normal" or theta_dist["name"] == "lognormal":
        pyro.sample(
            "theta",
            dist.MultivariateNormal(
                pyro.param("t mu", theta_dist["param"]["mu"] * torch.ones(n_models, dimension)),
                scale_tril=torch.tril(
                    pyro.param(
                        "t logstd",
                        torch.tensor(theta_dist["param"]["std"])
                        * torch.stack([log_cov_template] * n_models),
                    )
                ),
            ),
        )

    elif theta_dist["name"] == "beta":
        pyro.sample(
            "theta",
            dist.Beta(
                pyro.param(
                    "t alpha", theta_dist["param"]["alpha"] * torch.ones(n_models, dimension)
                ),
                pyro.param(
                    "t beta", theta_dist["param"]["beta"] * torch.ones(n_models, dimension)
                ),
            ),
        )
    else:
        raise TypeError(f"Theta distribution {theta_dist['name']} not supported.")


def get_model_guide(
    alpha_dist, theta_dist, alpha_transform, theta_transform, item_param_std, dimension=1, num_factors=3
):
    '''
    Method to define 3 parameter IRT model and guide given specifications for item discrimination [alpha]
    and responder ability [theta] parameter distributions and transfomations and standard deviations for item
    difficulty [beta] and log-guessing [log gamma] Gaussian distributions.

    Args:
        alpha_dist:      {`name`: distribution_name, `param`: distribution_dict}
                         Dictionary for the distribution type for the discrimation parameter [alpha].
                         distribution_dict is a dictionary of distribution parameters necessary for the
                         parameteric distribution defined by distribution_name.
        theta_dist:      {`name`: distribution_name, `param`: distribution_dict}
                         Dictionary for the distribution type for the ability parameter [theta].
                         distribution_dict is a dictionary of distribution parameters necessary for the
                         parameteric distribution defined by distribution_name.
        alpha_transform: Transformation applied to the discrimination parameter [alpha].
        theta_transform: Transformation aplied to the ability parameter [theta].
        item_params_std: Standard deviation for difficulty and guessing parameters.

    Returns:
        model:           3 parameter IRT model used for stochastic variation inference in Pyro
        guide:           3 parameter IRT guide used for stochastic variation inference in Pyro

    '''
    model = lambda obs: irt_model(
        obs,
        alpha_dist,
        theta_dist,
        alpha_transform=alpha_transform,
        theta_transform=theta_transform,
        item_params_std=item_param_std,
        dimension=dimension,
        num_factors=num_factors
    )
    guide = lambda obs: vi_posterior(obs, alpha_dist, theta_dist, dimension=dimension)

    return model, guide


def train(model, guide, data, optimizer, n_steps=500, weights=1, loss_type='weighted_elbo'):
    '''
    Method to fit 3 parameter IRT model parameters using stochastic variational inference.

    The method uses a weighted ELBO, where each item parameter's log-likelihood is weighted by the
    inverse of the item's dataset size. Use the default `weights=1` to use the standard ELBO.

    Args:
        model:      3 parameter IRT model
        guide:      3 parameter IRT guide
        data:       Numpy array of item responses
        optimizer:  Optimizer to use for stochastic variational inference
        n_steps:    Number of steps for fitting
        weights:    Weights to use for the weighted ELBO

    Returns:
        loss_track: List of weighted ELBO losses during the parameter fitting
    '''
    pyro.clear_param_store()
    print("train loss type: ", loss_type)
    if loss_type == 'weighted_elbo':
        print("Using weighted ELBO.")
        print(f'Max Weight: {max(weights):.3f}\nMin Weight: {min(weights):.3f}')
        svi_kernel = WeightedSVI(model, guide, optimizer, loss=Weighted_Trace_ELBO())
    elif loss_type == 'trace_elbo':
        print("Using weighted Trace ELBO.")
        svi_kernel = SVI(model, guide, optimizer, loss=Trace_ELBO())
    else:
        raise TypeError(f"{loss_type} not supported.")

    loss_track = []

    # do gradient steps
    data_ = torch.from_numpy(data.astype("float32"))
    t = tqdm(range(n_steps), desc="elbo loss", miniters=1, disable=False)
    for step in t:
        if loss_type == 'weighted_elbo':
            elbo_loss = svi_kernel.step(weights, data_)
        else:
            elbo_loss = svi_kernel.step(data_)

        t.set_description(f"elbo loss = {elbo_loss:.2f}")
        loss_track.append(elbo_loss)

    return loss_track


def get_response(file, ftype):
    if ftype == "csv":
        return pd.read_csv(file, index_col=0)
    else:
        raise KeyError(f"{ftype} reading not supported.")


def get_file_names(files_dir, ftype):
    file_list = []
    for root, dir, files in os.walk(files_dir):
        for file in files:
            if file.endswith(ftype):
                file_list.append(file)
        break
    return sorted(file_list)


def get_files(files_dir, ftype, taskset, verbose=False):
    responses = {}
    n_items = []
    data_names = []

    file_list = get_file_names(files_dir, ftype)
    for file in file_list:
        data_name = file[: -len("_irt_all_coded.csv")]

        if data_name in taskset:
            responses[data_name] = get_response(os.path.join(files_dir, file), ftype)
            n_items.append(responses[data_name].shape[1])
            data_names.append(data_name)

    if verbose:
        print(f"Missing:\n {set(taskset) - set(data_names)}")
    return data_names, responses, n_items


def subsample_responses(responses, sample_size, no_subsample, n_items, datasets):
    combined_responses = None

    if no_subsample:
        print("Use all examples:")
    else:
        print("Use sampling with sample size", sample_size)

    for idx, response in enumerate(responses.values()):
        min_items = n_items[idx] if sample_size > n_items[idx] else sample_size

        if no_subsample:
            sampled = response
            print(datasets[idx], "num test items: ", n_items[idx])
        else:
            sampled = response.sample(n=min_items, replace=False, axis=1)
            print(datasets[idx], "num test items: ", min_items)

        if combined_responses is None:
            combined_responses = sampled
            continue

        combined_responses = pd.merge(
            combined_responses, sampled, left_index=True, right_index=True,
        )

    return combined_responses


def get_distribution_params(distribution, verbose=False, std_overwrite=1.0):
    if verbose:
        print(f"Std Overwrite {std_overwrite:.2f}")

    if distribution == "normal":
        return {"mu": 0.0, "std": std_overwrite}
    elif distribution == "lognormal":
        return {"mu": 0.0, "std": std_overwrite}
    elif distribution == "beta":
        return {"alpha": 2.0, "beta": std_overwrite + 1}
    else:
        raise KeyError(f"Distribution type {distribution} not supported.")


def get_transform(transform):
    if transform == "identity":
        return lambda x: x
    elif transform == "positive":
        return lambda x: torch.log(1 + torch.exp(x))
    else:
        raise KeyError(f"Distribution type {transform} not supported.")


def set_seeds(seed):
    np.random.seed(seed)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)


def main(args):
    # Set seed
    set_seeds(args.seed)

    # Import response patterns
    data_names, responses, n_items = get_files(
        args.response_dir,
        args.response_type,
        args.datasets.split(","),
        verbose=args.verbose,
    )

    # Set weights for weighted ELBO
    if args.no_subsample:
        weights = sum([[1 / n] * n for n in n_items], [])
    else:
        weights = 1

    # Sample items
    min_items = min(n_items) if args.sample_size == -1 else args.sample_size
    combined_responses = subsample_responses(
        responses, min_items, args.no_subsample, n_items, data_names
    )
    import pdb; pdb.set_trace()
    if args.verbose:
        nl = "\n"
        tab = "\t"
        print(f"Extracted from {len(list(responses.keys()))} files")
        print(
            f"Collected response patterns for{nl+tab}{(nl+tab).join(list(responses.keys()))}"
        )
        print(f"Total number of items is {sum(n_items)}")
        print(f"Total combined items is {combined_responses.shape[1]}")
        print(f"Item Param Std is {args.item_param_std:.2f}")

    # Train model
    adam_params = {"lr": args.lr, "betas": (args.beta1, args.beta2)}
    optimizer = pyro.optim.AdamW(adam_params)

    model, guide = get_model_guide(
        {
            "name": args.discr,
            "param": get_distribution_params(
                args.discr, verbose=args.verbose, std_overwrite=args.alpha_std
            ),
        },
        {
            "name": args.ability,
            "param": get_distribution_params(args.ability, verbose=args.verbose),
        },
        get_transform(args.discr_transform),
        get_transform(args.ability_transform),
        args.item_param_std,
        args.dimension
    )

    print("loss type: ", args.loss_type)
    elbo_train_loss = train(
        model,
        guide,
        combined_responses.to_numpy(),
        optimizer,
        n_steps=args.steps,
        weights=weights,
        loss_type=args.loss_type
    )

    # Save parameters and sampled responses
    if args.no_subsample:
        exp_name = f"{args.lr}-num_factors-${args.num_factors}_alpha-{args.discr}-{args.discr_transform}-dim{args.dimension}_theta-{args.ability}-{args.ability_transform}_nosubsample_{args.item_param_std:.2f}_{args.alpha_std:.2f}"
    else:
        exp_name = f"{args.lr}-num_factors-${args.num_factors}alpha-{args.discr}-{args.discr_transform}-dim{args.dimension}_theta-{args.ability}-{args.ability_transform}_sample-{args.sample_size}_{args.item_param_std:.2f}_{args.alpha_std:.2f}"
    # out_dir = args.out_dir if args.out_dir != "" else os.path.join(".", "output")
    exp_path = args.out_dir if args.out_dir != "" else os.path.join('.', 'output', exp_name)
    os.makedirs(exp_path, exist_ok=True)
    print("last elbo: ", elbo_train_loss[-1])
    print("elbo losses: ", elbo_train_loss)
    pyro.get_param_store().save(os.path.join(exp_path, "params.p"))
    combined_responses.to_pickle(os.path.join(exp_path, "responses.p"))
    with open(os.path.join(exp_path, "train_elbo_losses.p"), 'wb') as f:
        pickle.dump(elbo_train_loss, f)
    print(f"Saved parameters and responses for {exp_name} in\n{exp_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        "--response_dir", help="directory containing responses", required=True
    )

    parser.add_argument(
        "--num_factors", help="number of factors in model", default=3, type=int
    )

    # Optional arguments
    parser.add_argument("--out_dir", default="", help="output directory")
    parser.add_argument(
        "--no_subsample", action="store_true", help="whether not to subsample responses"
    )
    parser.add_argument(
        "--sample_size",
        default=-1,
        help="number of items to sample per dataset, not used if --no_subsample is used.",
        type=int,
    )
    parser.add_argument(
        "--response_type", default="csv", help="response pattern file type", type=str
    )
    parser.add_argument("--seed", default=42, help="random seed", type=int)

    # Distribution arguments
    distribution_choices = ["normal", "lognormal", "beta"]
    transform_choices = ["identity", "positive"]

    parser.add_argument(
        "--diff",
        default="normal",
        help="difficulty (beta) distribution",
        choices=distribution_choices,
    )
    parser.add_argument(
        "--discr",
        default="normal",
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

    # Training arguments
    parser.add_argument(
        "--datasets",
        default="",
        help="comma separated string of datasets to include",
        type=str,
    )
    parser.add_argument(
        "--steps", default=500, help="number of training steps", type=int
    )
    parser.add_argument("--lr", default=1e-1, help="learning rate", type=float)
    parser.add_argument("--beta1", default=0.9, help="beta 1 for AdamW", type=float)
    parser.add_argument("--beta2", default=0.999, help="beta 2 for AdamW", type=float)
    parser.add_argument("--dimension", default=3, help="dimension of IRT", type=int)

    parser.add_argument("--loss_type", default="weighted_elbo", help="which loss to optimize to", type=str)

    # Tracking arguments
    parser.add_argument("--verbose", action="store_true", help="boolean for tracking")

    args = parser.parse_args()

    main(args)
