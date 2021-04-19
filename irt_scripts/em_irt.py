import argparse
import os
import pickle

import torch
from torch.distributions.bernoulli import Bernoulli
from variational_irt import get_files, set_seeds, subsample_responses


def write_pickle(file, fname, out_dir):
    with open(os.path.join(out_dir, fname), 'wb') as f:
        pickle.dump(file, f)


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
                a[:, None, :],
                (b[:, None, :] - theta[None, :, :]).transpose(2, 1) # nitems x ndims x nresp
            )
        ).squeeze()
    else:
        raise KeyError(f'{nparams} not supported')


def get_nll(responses, thetas, itemparams):
    probs = probfunc(thetas, itemparams).T
    rv = Bernoulli(probs)
    return -rv.log_prob(responses).sum()


def sgd(responses, thetas, itemparams, steps, lr=0.1, mode='e', return_losses=False):
    if mode =='e':
        optimizer = torch.optim.SGD([thetas], lr=lr)
    elif mode =='m':
        optimizer = torch.optim.SGD([itemparams], lr=lr)
    else:
        raise KeyError(f'Mode {mode} not supported.')

    losses = []
    for s in range(steps):

        optimizer.zero_grad()
        loss = get_nll(responses, thetas, itemparams)
        losses.append(loss)
        loss.backward()
        optimizer.step()

    if return_losses:
        return losses


def fit_em_irt(
        responses,
        device,
        max_steps=1000,
        tol=1e-14,
        ntol=5,
        ndims=1,
        nparams=1,
        sgd_steps=50,
        sgd_lr=0.01,
        init_scale=1e2,
        verbose=False,
        verbose_steps=50,
):
    # EM algorithm for fitting IRT

    nresp, nitems = responses.shape

    if verbose:
        print(f'Using device {device}')

    thetas = torch.normal(0, init_scale, (nresp, ndims), requires_grad=True, device=device)
    itemparams = torch.normal(0, init_scale, (nitems, nparams, ndims), requires_grad=True, device=device)

    losses = [get_nll(responses, thetas, itemparams).cpu().item()]
    tol_count, steps = 0, 0
    while tol_count < ntol and steps < max_steps:
        prev_thetas, prev_itemparams = thetas.clone(), itemparams.clone()

        # E step SGD
        sgd(responses, thetas, itemparams, sgd_steps, lr=sgd_lr, mode='e')

        # M step SGD
        sgd(responses, thetas, itemparams, sgd_steps, lr=sgd_lr, mode='m')

        losses.append(
            get_nll(responses, thetas, itemparams).cpu().item()
        )

        # increment tolerance count if not above `tol`
        if max(
                torch.norm(prev_thetas - thetas, p=float('inf')),
                torch.norm(prev_itemparams - itemparams, p=float('inf')),
        ) < tol:
            tol_count += 1

        steps += 1

        if verbose and steps % verbose_steps == 0:
            print(f'Step {steps}')

    if verbose:
        print(f'Finished in {steps} steps.')

    orders = ['a', 'b', 'g']

    return thetas.cpu(), itemparams.cpu(), orders[:nparams+1], losses

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

    # Sample items
    min_items = min(n_items) if args.sample_size == -1 else args.sample_size
    combined_responses = subsample_responses(
        responses, min_items, args.no_subsample, n_items, data_names
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    thetas, itemparams, order, losses = fit_em_irt(
        torch.tensor(combined_responses.to_numpy(dtype="float32")).to(device),
        device,
        nparams=args.nparams,
        max_steps=args.max_steps,
        ndims=args.ndims,
        init_scale=args.init_scale,
        verbose=args.verbose,
        sgd_lr=args.lr,
    )

    out_dir = os.path.join('.', args.out_dir) if args.out_dir == 'test_params' else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    write_pickle(thetas, 'thetas.p', out_dir)
    write_pickle(itemparams, 'item_params.p', out_dir)
    write_pickle(order, 'params_order.p', out_dir)
    write_pickle(losses, 'losses.p', out_dir)

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

    parser.add_argument('--max_steps', default=1000, help='maximum number of EM steps', type=int)
    parser.add_argument('--ndims', default=1, help='number of dimensions', type=int)
    parser.add_argument('--init_scale', default=10, help='number of dimensions', type=int)
    parser.add_argument('--lr', default=0.01, help='optimizer learning rate', type=float)
    parser.add_argument('--nparams', default=1, help='optimizer learning rate', type=int)

    parser.add_argument("--seed", default=42, help="random seed", type=int)
    parser.add_argument(
        "--datasets",
        default="",
        help="comma separated string of datasets to include",
        type=str,
    )

    parser.add_argument(
        "--no_subsample", action="store_true", help="whether not to subsample responses"
    )
    parser.add_argument(
        "--sample_size",
        default=-1,
        help="number of items to sample per dataset, not used if --no_subsample is used.",
        type=int,
    )

    parser.add_argument("--verbose", action="store_true", help="boolean for tracking")

    args = parser.parse_args()

    main(args)
