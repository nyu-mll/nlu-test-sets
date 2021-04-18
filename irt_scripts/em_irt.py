import argparse
import os
import pickle

import torch
from torch.distributions.bernoulli import Bernoulli
from variational_irt import get_files, set_seeds

def sigmoid(x):
    return 1/(1+torch.exp(-x))

def probfunc(theta, itemparams):
    # calculates probabilites using model abilities and item parameters
    nitems, nparams, ndims = itemparams.shape

    if nparams == 1:
        # 1PL
        b = itemparams
        return sigmoid(
            (b - theta[None, :, :]).sum(axis=2)
        ).squeeze()
    elif nparams == 2:
        # 2PL
        a = itemparams[:, 0, :]
        b = itemparams[:, 1, :]

        return sigmoid(
            torch.matmul(
                a,
                (b - theta[None, :, :]).transpose(2, 1) # nitems x ndims x nresp
            )
        ).squeeze()
    else:
        raise KeyError(f'{nparams} not supported')

def sgd(responses, thetas, itemparams, steps, lr=0.1, mode='e', return_losses=False):
    if mode =='e':
        optimizer = torch.optim.SGD(thetas, lr=lr)
    elif mode =='m':
        optimizer = torch.optim.SGD(itemparams, lr=lr)
    else:
        raise KeyError(f'Mode {mode} not supported.')

    losses = []
    for s in range(steps):

        optimizer.zero_grad()
        probs = probfunc(thetas, itemparams)
        rv = Bernoulli(probs)
        loss = -rv.log_prob(responses)
        losses.append(loss)
        loss.backward()
        optimizer.step()

    if return_losses:
        return losses

def fit_em_irt(
        responses,
        max_steps=1000,
        tol=1e-14,
        ntol=5,
        ndims=1,
        nparams=1,
        sgd_steps=500,
        init_scale=1e2,
):
    # EM algorithm for fitting IRT

    nresp, nitems = responses.shape

    thetas = torch.randn((nresp, ndims))*init_scale
    itemparams = torch.randn((nitems, nparams, ndims))*init_scale

    tol_count, steps = 0, 0
    while tol_count < ntol and steps < max_steps:
        prev_thetas, prev_itemparams = thetas.clone(), itemparams.clone()

        # E step SGD
        sgd(responses, thetas, itemparams, sgd_steps, mode='e')

        # M step SGD
        sgd(responses, thetas, itemparams, sgd_steps, mode='m')

        # increment tolerance count if not above `tol`
        if max(
                torch.norm(prev_thetas - thetas, p='inf'),
                torch.norm(prev_itemparams, itemparams, p='inf')
        ) < tol:
            tol_count += 1

        steps += 1

    orders = ['a', 'b', 'g']

    return thetas, itemparams, orders[:nparams]

def write_pickle(file, fname, out_dir):
    with open(os.path.join(out_dir, fname), 'wb') as f:
        pickle.dump(file, f)

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

    thetas, itemparams, order = fit_em_irt(
        responses
    )

    out_dir = os.path.join('.', args.out_dir) if args.out_dir == 'test_params' else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    write_pickle(thetas, 'thetas.p', out_dir)
    write_pickle(itemparams, 'item_params.p', out_dir)
    write_pickle(order, 'params_order.p', out_dir)

    if args.verbose:
        print(
            f'='*40 + f' Complete ' + f'='*40 +
            f'\nSaved parameters to: {out_dir}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        "--response_dir", help="directory containing responses", required=True
    )

    # Optional arguments
    parser.add_argument("--out_dir", default="test_params", help="output directory")
    parser.add_argument(
        "--response_type", default="csv", help="response pattern file type", type=str
    )
    parser.add_argument("--seed", default=42, help="random seed", type=int)
    parser.add_argument(
        "--datasets",
        default="",
        help="comma separated string of datasets to include",
        type=str,
    )

    parser.add_argument("--verbose", action="store_true", help="boolean for tracking")

    args = parser.parse_args()

    main(args)
