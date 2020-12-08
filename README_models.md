# Model Training and Generating Responses

We use `jiant v2.1.1` to train our models. To learn more about `jiant`, please refer to the [documentation](https://github.com/nyu-mll/jiant/tree/v2.1.1).

## Models
We train **18 different models** in total (provided by Huggingface): `albert-xxlarge-v2`, `roberta-large`, `roberta-base`, `bert-large-uncased`, `bert-base-uncased`, `xlm-roberta-large`, and 12 `miniBERTas` models ([Warstadt et al., 2020](https://arxiv.org/pdf/2010.05358.pdf); [Zhang et al., 2020](https://arxiv.org/abs/2011.04946)) listed [here](https://huggingface.co/nyu-mll).

# Steps for training the models

## Setting up `jiant`

- Clone jiant and checkout `IRT_experiments` branch.
```
mkdir jiant_exp
cd jiant_experiments
git clone https://github.com/nyu-mll/jiant.git
git checkout IRT_experiments
```

- Install `jiant` from source by following  the steps [here](https://github.com/nyu-mll/jiant/tree/IRT_experiments#installation).
```
cd jiant
pip install -r requirements.txt
cd ..

# Add the following to your .bash_rc or .bash_profile 
export PYTHONPATH=/path/to/jiant_exp/jiant:$PYTHONPATH
```

## Downloading models

We first need to download and cache the models from Huggingface. We will save our models and caches in `/path/to/jiant_exp/experiments` folder. To download all 18 models used in our experiment, run:

```
mkdir -p experiments
source jiant/irt_scripts/download_all_models.sh $(pwd)
download_all_models
```

You can also download individual models using the `download_models.sh` script. For example, to download the `roberta-large` model:

```
MODEL_TYPE=roberta-large
source jiant/irt_scripts/download_all_models.sh $(pwd)
download_model $MODEL_TYPE
```

## Downloading datasets
Please email `c.vania at nyu dot edu` for custom datasets used in our project.

## Preparing datasets

- We train each model on *28 datasets* listed in the paper. 
- First, copy the `configs.zip` from here and unzip it under `/path/to/jiant_exp/experiments/tasks`
- We need to change the data paths in all the configs to your `jiant` directory (current working directory). To do this, run:
```
find ./experiments/tasks/configs/ -type f -exec sed -i -e 's/pathto/$(pwd)/g' {} \;
```

- We need to tokenize and cache the tokenized data. The script for preprocessing is in `jiant/irt_scripts/run_preprocess.sh`.

- To preprocess all 28 datasets using a model, run:
```
source jiant/irt_scripts/run_preprocess.sh

MODEL_TYPE=roberta-large
prepare_all_tasks $MODEL_TYPE
```
To preprocess individual datasets, use `preprocess_task` function from `jiant/irt_scripts/run_preprocess.sh`.
For example, to preprocess `copa` using `roberta-large` tokenizer, run:
```
source jiant/irt_scripts/run_preprocess.sh
preprocess_task roberta-large copa
```

## Hyperparameter Tuning
Please skip this step if you don't want to perform hyperparmeter tuning. 
```
source jiant/irt_scripts/taskmaster_hyperparemters.sh
tune_hyperparameters $MODEL_TYPE $TASKMASTER_TASKS
```

## Training a model
To train the model using our best hyperparameter config, run:
```
source jiant/irt_scripts/taskmaster_hyperparemters.sh
bash train_all_models $MODEL_TYPE $DATASET
```

## Generating Responses
To generate responses on a dataset using all the models, run:
```
python jiant/irt_scripts/call_predict.py $MODEL_TYPE $DATASET
```
Make sure to change the `output path` in `jiant/irt_scripts/call_predict.py`.


