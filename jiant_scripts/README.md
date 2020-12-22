# Model Training and Generating Responses

The following README describes step-by-step how to run experiments for one or more datasets and how to generate model responses using the trained models. We use `jiant v2.1.1` to train our models. To learn more about `jiant`, please refer to the [documentation](https://github.com/nyu-mll/jiant/tree/v2.1.1).

## Models

We train **18 different models** in total (provided by Huggingface): `albert-xxlarge-v2`, `roberta-large`, `roberta-base`, `bert-large-uncased`, `bert-base-uncased`, `xlm-roberta-large`, and 12 `miniBERTas` models ([Warstadt et al., 2020](https://arxiv.org/pdf/2010.05358.pdf); [Zhang et al., 2020](https://arxiv.org/abs/2011.04946)) listed [here](https://huggingface.co/nyu-mll). You can find the list of model names in `models.txt`.

## Training Models

### Setting up `jiant`

- Clone jiant and checkout `IRT_experiments` branch.
```
mkdir jiant_exp
cd jiant_exp
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

### Downloading pretrained models

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

### Datasets

If you want to run experiments for a dataset that is supported by `jiant`, you can download it using the following command:
```
python jiant/scripts/download_data/runscript.py \
    download \
    --tasks task_name \
    --output_path /path/to/exp/tasks
```
Note that if you want to use the datasets that we used in the paper, some of them do not have publicly available test labels. If you are interested to use our customized datasets, please contact one of the paper's authors. Additionally, if your dataset doesn't have publicly available labeled test set, you would need to create your own custom split.


### Preparing datasets

- First, you need to write a task config for your dataset. Examples of task config that we used can be found in [`configs.zip`](https://github.com/nyu-mll/nlu-test-sets/tree/main/jiant_task_data). Put the (unzip) task configs under `/path/to/jiant_exp/experiments/tasks`

- After that, we need to change the data paths in all the configs to your `jiant` directory (current working directory). To do this, run:
```
find ./experiments/tasks/configs/ -type f -exec sed -i -e 's/pathto/$(pwd)/g' {} \;
```

- Next, we need to tokenize and cache the tokenized data. The script for preprocessing is in `jiant/irt_scripts/run_preprocess.sh`. To preprocess all datasets using a model, run:
```
source jiant/irt_scripts/run_preprocess.sh

MODEL_TYPE=roberta-large
prepare_all_tasks $MODEL_TYPE
```

- Alternatively, to preprocess individual datasets, you can use `preprocess_task` function from `jiant/irt_scripts/run_preprocess.sh`. For example, to preprocess `copa` using `roberta-large` tokenizer, run:
```
source jiant/irt_scripts/run_preprocess.sh
preprocess_task roberta-large copa
```

### Hyperparameter Tuning
You can skip this step if you don't want to perform hyperparmeter tuning. If you are adding a new dataset, you need to declare the training size in `jiant/irt_scripts/taskmaster_hyperparmeters.sh`.
```
source jiant/irt_scripts/training_scripts.sh
tune_hyperparameters $MODEL_TYPE $TASKMASTER_TASKS
```

### Training a model on all tasks using our best config
If you want to replicate our experiments, the scripts for model training can be found in `jiant/irt_scripts/training_scripts.sh`. We performed the hyperparameter tuning using `roberta-large` model to find the best training config for each task. To train the models using our best config:
```
source jiant/irt_scripts/training_scripts.sh

MODEL_TYPE=roberta-large
train_best_configs $MODEL_TYPE
```

### Training a model
Alternatively, you can also train an individual model:
```
source jiant/irt_scripts/training_scripts.sh
run_training $MODEL_TYPE $DATASET $CONFIG_NO
```


## Generating Responses
To generate responses on a dataset using all the models, run:
```
python jiant/irt_scripts/call_predict.py $MODEL_TYPE $DATASET $(pwd)
```
This will generate predictions and responder accuracies.

To combine the generated responses into a csv file (which will be used as an input for IRT analysis), run
```
python jiant/irt_scripts/postprocess_predictions.py $(pwd)
```


