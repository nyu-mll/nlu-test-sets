import os
import pickle
import copy

import pandas as pd
import seaborn as sns
import numpy
import torch
import scipy
import scipy.stats

import pyro
import pyro.infer
import pyro.infer.mcmc
import pyro.distributions as dist
import torch.distributions.constraints as constraints
from tqdm.auto import tqdm

import matplotlib.pyplot as plot
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from multi_virt_v2 import *

import warnings
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

import sys
dim = sys.argv[1]
base_dir=sys.argv[2]
saved_param_dir=sys.argv[3]
file_name=sys.argv[4]
print(file_name)

exp_dir = os.path.join(base_dir, saved_param_dir, file_name)
p = 0.95

def sigmoid(x):
    return 1./(1.+torch.exp(-x))

def icc_best_deriv(alpha, beta, theta, model_names, gamma=None, col='mean'):
    '''
    Method to calculate the locally estimated headroom (LEH) score, defined as
    the derivative of the item characteristic curve w.r.t. the best performing model.
    
    Args:
        alpha:       DataFrame of discrimination parameter statistics for each item.
        beta:        DataFrame of difficulty parameter statistics for each item.
        theta:       DataFrame of ability parameter statistics for each responder.
        model_names: List of responder names.
        gamma:       DataFrame of guessing parameter statistics for each item.
        col:         DataFrame column name to use for calculating LEH scores.
    
    Returns:
        scores:      LEH scores for each item.    
    '''
    best_idx, best_value = theta[col].argmax(), theta[col].max()
    print(f'Best model: {model_names[best_idx]}\n{best_value}')
    
    a, b = torch.tensor(alpha[col].values), torch.tensor(beta[col].values)
    
    #logits = (a*(best_value-b))
    logits = (torch.matmul(a,best_value.T) + b).T
    sigmoids = sigmoid(logits)
    scores = sigmoids*(1.-sigmoids)*a
    
    print(f'No gamma: {scores.mean()}')
    if not gamma is None:
        g = torch.tensor(gamma[col].apply(lambda x: x.item()).values)
        scores = (1.-g)*scores
        print(f'With gamma: {scores.mean()}')
    
    return scores      
    
    
def get_model_guide(alpha_dist, theta_dist, alpha_transform, theta_transform):
    model = lambda obs: irt_model(obs, alpha_dist, theta_dist, alpha_transform = alpha_transform, theta_transform = theta_transform)
    guide = lambda obs: vi_posterior(obs, alpha_dist, theta_dist)
    
    return model, guide

def get_data_accuracies(data, verbose = False, get_cols = False):
    '''
    Method to reformat `data` and calculate item and responder accuracies.
    
    Args:
        data:                DataFrame of item responses.
        verbose:             Boolean value of whether to print statements.
        get_cols:            Boolean value of whether to return original column
                             values of `data`.
        
    Returns:
        new_data:            Reformatted `data`, dropping first column.
        accuracies:          Accuracy for each responder across examples.
        example_accuracies:  Accuracy for each example across responders.
        data.columns.values: Returns only if `get_cols` is True. Original column
                             values of `data`.
    '''
    new_data = numpy.array(data)
    new_data = new_data[:,1:]
    
    model_names = dict(data['userid'])
    accuracies = new_data.mean(-1)
    example_accuracies = new_data.mean(0)
    
    if verbose:
        print('\n'.join([f'{name}: {acc}' for name, acc in zip(model_names.values(),accuracies)]))
    
    if get_cols:
        return new_data, accuracies, example_accuracies, data.columns.values
    else:
        return new_data, accuracies, example_accuracies


def get_stats_CI(params, p=0.95, dist='normal'):
    '''
    Method to calculate lower and upper quantiles defined by `p`, mean, and variance of `param`
    
    Args:
        params: Dictionary of distribution parameters for each item keyed according to the 
                parametric distribution defined by `dist`.
        p:      Percent of distribution covered by the lower and upper interval values for each
                parameter.
        dist:   Name of parametric distribution
    
    Returns:
        return: {
            'lower': Lower interval values of each parameter,
            'upper': Upper interval values of each parameter,
            'mean' : Mean of each parameter,
            'var'  : Variance of each parameter
        }
    '''
    stats = {}
    if dist == 'normal':
        L,U = scipy.stats.norm.interval(p,loc=params['mu'], scale=torch.exp(params['logstd']))
        M,V = scipy.stats.norm.stats(loc=params['mu'], scale=torch.exp(params['logstd']))
    elif dist == 'log-normal':
        L,U = scipy.stats.lognorm.interval(p, s=torch.exp(params['logstd']), scale=torch.exp(params['mu']))
        M,V = scipy.stats.lognorm.stats(s=torch.exp(params['logstd']), scale=torch.exp(params['mu']))
    elif dist == 'beta':
        L,U = scipy.stats.beta.interval(p,a=params['alpha'], b=params['beta'])
        M,V = scipy.stats.beta.stats(a=params['alpha'], b=params['beta'])
    else:
        raise TypeError(f'Distribution type {dist} not supported.')
    
    return {
        'lower':[L],
        'upper':[U],
        'mean':[M],
        'var':[V],
    }


def get_plot_stats(exp_dir, alpha_dist, theta_dist, transforms, p = 0.95):
    '''
    Method to return plotting statistics for 3 parameter IRT model parameters.
    
    Args:
        exp_dir:          Path to 3 parameter IRT parameters and responses.
        alpha_dist:       Name of the item discrimination [a] distribution.
        theta_dist:       Name of the responder ability [t] distribution.
        transforms:       Dictionary of transformations to apply to each parameter type
                          where keys are parameter names and values are functions.
        p:                Percent of distribution covered by the lower and upper interval 
                          values for each parameter.
    
    Returns:
        param_plot_stats: Dictionary of parameter plot statistics where keys are parameter
                          names and values are plot statistics dictionaries as defined by
                          get_stats_CI().
    '''
    param_dists = {
        'a':alpha_dist,
        'b':'normal',
        'g':'normal',
        't':theta_dist,
    }

    dist_params = {
        'normal':['mu', 'logstd'],
        'log-normal':['mu', 'logstd'],
        'beta':['alpha', 'beta'],
    }

    pyro.clear_param_store()
    pyro.get_param_store().load(os.path.join(exp_dir, 'params.p'))

    with torch.no_grad():
        pyro_param_dict = dict(pyro.get_param_store().named_parameters())
    
    # get stats for plotting
    param_plot_stats = {}

    for param, param_dist in param_dists.items():
        temp_params = dist_params[param_dist]

        for idx, (p1_orig, p2_orig) in enumerate(zip(pyro_param_dict[f'{param} {temp_params[0]}'], pyro_param_dict[f'{param} {temp_params[1]}'])):
            p1, p2 = p1_orig.detach(), p2_orig.detach()
            
            temp_stats_df = pd.DataFrame.from_dict(
                get_stats_CI(
                    params = {
                        temp_params[0]:p1,
                        temp_params[1]:p2,
                    },
                    p=p,
                    dist = param_dist,
                )
            )
            
            temp_stats_df = temp_stats_df.applymap(transforms[param])
        
            if idx == 0:
                param_plot_stats[param] = temp_stats_df
            else:
                param_plot_stats[param] = param_plot_stats[param].append(temp_stats_df, ignore_index = True)
    
    return param_plot_stats


def sign_mult(df1, df2):
    newdf = copy.deepcopy(df2)
    
    for idx, row in df1.iterrows():
        if numpy.sign(row['mean']) < 0:
            newdf.loc[idx,'mean'] = -1*newdf.loc[idx,'mean']
            newdf.loc[idx,'lower'] = -1*newdf.loc[idx,'upper']
            newdf.loc[idx,'upper'] = -1*newdf.loc[idx,'lower']
    
    return newdf



def get_diff_by_set(diffs, item_ids):
    diff_by_set = {}
    id_split = '_'

    max_diff = -1e6
    min_diff = 1e6
    
    for idx, diff in enumerate(diffs):
        set_name = item_ids[idx].split(id_split)[0]

        if set_name in diff_by_set.keys():
            diff_by_set[set_name].append(diff)
        else:
            diff_by_set[set_name] = [diff]
            
        if diff < min_diff:
            min_diff = diff
            
        if diff > max_diff:
            max_diff = diff
    
    return diff_by_set, min_diff, max_diff


"""
datasets="boolq,cb,commonsenseqa,copa,cosmosqa,hellaswag,rte,snli,wic,qamr,arct,mcscript,mctaco,mutual,mutual-plus,quoref,socialiqa,squad-v2,wsc,mnli,mrqa-nq,newsqa,abductive-nli,arc-easy,arc-challenge,piqa,quail,winogrande,anli"
data_names, responses, n_items = get_files(
    os.path.join(base_dir, 'data'),
    "csv",
    set(datasets.split(','))
)
task_metadata = pd.read_csv(os.path.join(base_dir,'irt_scripts','task_metadata.csv'))
task_metadata.set_index("jiant_name", inplace=True)
task_list = [x for x in task_metadata.index if x in data_names]

total = 0
task_name = []
task_format = []

for tname, size in zip(data_names, n_items):
    name = task_metadata.loc[tname]['taskname']
    total += size
    task_name += [name for _ in range(size)]
    task_format += [task_metadata.loc[tname]['format'] for _ in range(size)]
    
task_name = pd.DataFrame(task_name, columns=['task_name'])
task_format = pd.DataFrame(task_format, columns=['format'])
task_name_format = pd.concat([task_name, task_format], axis=1)





combined_responses = pd.read_pickle(os.path.join(exp_dir, 'responses.p')).reset_index()
data, accuracies, example_accuracies = get_data_accuracies(combined_responses)
column_names = combined_responses.columns[1:]
"""

print("Saving plot stats")
load_from_cache = False

# distribution and transformation
alpha_dist = 'log-normal'
alpha_transf = 'standard'
theta_dist = 'normal'
theta_transf = 'standard'


select_ts = {
    'standard':lambda x:x,
    'positive':lambda x:torch.log(1+torch.exp(torch.tensor(x))),
    'sigmoid':lambda x:sigmoid(torch.tensor(x)),
}

transforms = {
    'a':select_ts[alpha_transf],
    'b':select_ts['standard'],
    'g':select_ts['sigmoid'],
    't':select_ts[theta_transf],
}

if load_from_cache:
    param_plot_stats = {}

    for key in transforms.keys():
        with open(os.path.join(exp_dir, 'plot_stats_pickles', f'{key}.p'), 'rb') as f:
            param_plot_stats[key] = pickle.load(f)
else:
    param_plot_stats = get_plot_stats(
        exp_dir,
        alpha_dist,
        theta_dist,
        transforms,
        p = 0.95
    )
    plot_dir = os.path.join(exp_dir, 'plot_stats_pickles')
    os.makedirs(plot_dir, exist_ok=True)
    for key, value in param_plot_stats.items():
        with open(os.path.join(exp_dir, 'plot_stats_pickles', f'{key}.p'), 'wb') as f:
            pickle.dump(value, f)


print("saved plot stats in : ", plot_dir)




