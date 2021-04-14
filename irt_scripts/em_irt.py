import numpy as np
from numpy.linalg import norm

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_probfunc(itemparams):
    # returns function taking abilities, theta, and returns nitems x nresp of probabilities

    nitems, nparams, ndims = itemparams.shape

    if nparams == 1:
        # 1PL
        b = itemparams
        return lambda theta: sigmoid(
            (b - theta[None, :, :]).sum(axis=2)
        ).squeeze()
    elif nparams == 2:
        # 2PL
        a = itemparams[:, 0, :]
        b = itemparams[:, 1, :]

        return lambda theta: sigmoid(
            np.matmul(
                a,
                (b - theta[None, :, :]).swapaxes(2, 1) # nitems x ndims x nresp
            )
        ).squeeze()
    else:
        raise KeyError(f'{nparams} not supported')

def e_step(responses, probfunc, estep_particles):
    # perform estep using MC estimation
    nresp, nitems = responses.shape

    # TODO

    pass

def m_step(responses, thetas, mstep_steps):
    # TODO
    
    pass

def fit_em_irt(
        responses,
        max_steps=1000,
        tol=1e-14,
        ntol=5,
        ndims=1,
        nparams=1,
        estep_particles=1000,
        mstep_steps=500,
):
    # EM algorithm for fitting IRT

    nresp, nitems = responses.shape

    thetas = np.random.standard_normal((nresp, ndims))
    itemparams = np.random.standard_normal((nitems, nparams, ndims))
    prev_thetas, prev_itemparams = None, None

    tol_count, steps = 0,0
    while tol_count < ntol and steps < max_steps:
        prev_thetas, prev_itemparams = thetas, itemparams

        probfunc = get_probfunc(itemparams)
        thetas = e_step(responses, probfunc, estep_particles)
        itemparams = m_step(responses, thetas, mstep_steps)

    pass
