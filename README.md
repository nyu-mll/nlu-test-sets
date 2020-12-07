# nlu-test-sets

This is a repository for the paper "Quoref is Still Effective, SNLI is Not: Comparing Test Sets with Item Response Theory".

## Requirements

For training the models and generating new model responses using `jiant` (we use v2.1.1), please refer to the related [README](https://github.com/nyu-mll/nlu-test-sets/blob/main/README_models.md).

For IRT analysis, we use the following packages:
- pyro-api==0.1.2
- pyro-ppl==1.4.0
- matplotlib==3.2.2
- seaborn==0.11.0


## IRT Analysis

The `data` directory contains model responses for the 28 datasets described in the paper. The `params` directory consists of estimated parameters based on the model responses. You can then simply run the `plot_virt.ipynb` to generate some plots.


## Analyzing New Test Set(s)

### Generating Model Responses

To run the same analysis for new test set(s), first you would need to train each dataset using 18 Transformer models used in the paper. We use `jiant v2.1.1` for all of our experiments. The scripts for hyperparameter tuning, storing checkpoints, and evaluation can be found in the [IRT_experiment](https://github.com/nyu-mll/jiant/tree/IRT_experiments) branch. Please refer to the provided [readme]() if you want to run a new experiment and generate the corresponding model responses.


### Fitting an IRT model

After obtaining the model responses for each test set, add them into the `data` folder. After that, edit the `irt_scripts/estimate_irt_params.sh` by adding the new test sets into `TASKS` list. You also need to modify the `PATH` so that it will refer to your working directory.

You can use the following command to estimate parameters:
```
bash irt_scripts/estimate_irt_params.sh
```
The script will generate some parameters which will be store in the `params` directory. There are two files, `params.p` and `responses.p`.

After that, you can run the same notebook to generate plots.
