# Comparing Test Sets with IRT

This is a repository for the paper ["Comparing Test Sets with Item Response Theory"](https://arxiv.org/abs/2106.00840), to appear at ACL 2021.

## Requirements

For training the models and generating new model responses using `jiant` (we use v2.1.1), please refer to the related [README](https://github.com/nyu-mll/nlu-test-sets/blob/main/README_models.md).

For IRT analysis, use Python 3 (we use Python 3.7) and install the required packages.

```
git clone https://github.com/nyu-mll/nlu-test-sets.git
cd nlu-test-sets
pip install -r requirements.txt
```


## IRT Analysis

The `data` directory contains model responses for the 29 datasets described in the paper. The `params` directory consists of estimated parameters based on the model responses. You can then simply run the `plot_virt.ipynb` to generate some plots.


## Analyzing New Test Set(s)

### Generating Model Responses

To run the same analysis for new test set(s), first you would need to train each dataset using 18 Transformer models used in the paper. We use `jiant v2.1.1` for all of our experiments. The scripts for hyperparameter tuning, storing checkpoints, and evaluation can be found in the [IRT_experiment](https://github.com/nyu-mll/jiant/tree/IRT_experiments) branch. Please refer to the provided [readme](https://github.com/nyu-mll/nlu-test-sets/tree/main/jiant_scripts) if you want to run a new experiment and generate the corresponding model responses.


### Fitting an IRT model

After obtaining the model responses for each test set, add them into the `data` folder. After that, edit the `irt_scripts/estimate_irt_params.sh` by adding the new test sets into the `TASKS` list. You also need to modify the `PATH` so that it will refer to your working directory.

You can use the following command to estimate parameters:
```
bash irt_scripts/estimate_irt_params.sh
```
The script will generate some parameters which will be store in the `params` directory. There are two files, `params.p` and `responses.p`.

After that, you can run the same notebook to generate plots.
