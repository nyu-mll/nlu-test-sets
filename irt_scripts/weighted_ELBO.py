from __future__ import absolute_import, division, print_function
import torch
import warnings
import pyro
import pyro.poutine as poutine
from pyro.util import check_model_guide_match, check_site_shape, torch_isnan

import pyro.ops.jit
from pyro.distributions.util import is_identically_zero, scale_and_mask
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import (
    MultiFrameTensor,
    get_plate_stacks,
    is_validation_enabled,
    torch_item,
)
from pyro.util import check_if_enumerated, warn_if_nan


from pyro.infer import SVI


class Weighted_Trace_ELBO(ELBO):
    def _get_trace(self, model, guide, args, kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, args, kwargs
        )
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace

    def _differentiable_loss_particle(self, model_trace, guide_trace, obs_weights):
        elbo_particle = 0
        surrogate_elbo_particle = 0
        log_r = None

        obs_weights = torch.Tensor(obs_weights)
        # compute elbo and surrogate elbo

        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                if name != "theta":
                    weights = obs_weights
                else:
                    weights = 1.0


                elbo_particle = elbo_particle + torch_item(
                    scale_and_mask(site["log_prob"], weights).sum()
                )
                surrogate_elbo_particle = (
                    surrogate_elbo_particle
                    + scale_and_mask(site["log_prob"], weights).sum()
                )

        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                log_prob, score_function_term, entropy_term = site["score_parts"]

                if name != "theta":
                    weights = obs_weights
                else:
                    weights = 1.0

                elbo_particle = elbo_particle - torch_item(
                    scale_and_mask(site["log_prob"], weights).sum()
                )

                if not is_identically_zero(entropy_term):
                    surrogate_elbo_particle = (
                        surrogate_elbo_particle
                        - scale_and_mask(entropy_term, weights).sum()
                    )

                if not is_identically_zero(score_function_term):
                    if log_r is None:
                        log_r = _compute_log_r(model_trace, guide_trace)
                    site = log_r.sum_to(site["cond_indep_stack"])
                    surrogate_elbo_particle = (
                        surrogate_elbo_particle
                        + scale_and_mask((site * score_function_term), weights).sum()
                    )

        return -elbo_particle, -surrogate_elbo_particle

    def differentiable_loss(self, model, guide, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the model and guide parameters
        """
        loss = 0.0
        surrogate_loss = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            loss_particle, surrogate_loss_particle = self._differentiable_loss_particle(
                model_trace, guide_trace
            )
            surrogate_loss += surrogate_loss_particle / self.num_particles
            loss += loss_particle / self.num_particles
        warn_if_nan(surrogate_loss, "loss")
        return loss + (surrogate_loss - torch_item(surrogate_loss))

    def loss_and_grads(self, model, guide, weights, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        loss = 0.0
        # grab a trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):

            loss_particle, surrogate_loss_particle = self._differentiable_loss_particle(
                model_trace, guide_trace, weights
            )
            loss += loss_particle / self.num_particles

            # collect parameters to train from model and guide
            trainable_params = any(
                site["type"] == "param"
                for trace in (model_trace, guide_trace)
                for site in trace.nodes.values()
            )

            if trainable_params and getattr(
                surrogate_loss_particle, "requires_grad", False
            ):
                surrogate_loss_particle = surrogate_loss_particle / self.num_particles
                surrogate_loss_particle.backward(retain_graph=self.retain_graph)
        warn_if_nan(loss, "loss")
        return loss

    def loss(self, model, guide, weights, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0

        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            weight_model_log_p = model_trace.compute_log_prob()
            guide_log_p = guide_trace.compute_log_prob()
            elbo_particle = torch_item(weight_model_log_p.sum()) - torch_item(
                guide_log_p.sum()
            )
            elbo += elbo_particle / self.num_particles

        loss = -elbo

        if torch_isnan(loss):
            warnings.warn("Encountered NAN loss")
        return loss


class WeightedSVI(SVI):
    def step(self, weights, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Take a gradient step on the loss function (and any auxiliary loss functions
        generated under the hood by `loss_and_grads`).
        Any args or kwargs are passed to the model and guide
        """
        # get loss and compute gradients
        with poutine.trace(param_only=True) as param_capture:
            loss = self.loss_and_grads(self.model, self.guide, weights, *args, **kwargs)

        params = set(
            site["value"].unconstrained() for site in param_capture.trace.nodes.values()
        )

        # actually perform gradient steps
        # torch.optim objects gets instantiated for any params that haven't been seen yet
        self.optim(params)

        # zero gradients
        pyro.infer.util.zero_grads(params)

        return torch_item(loss)
