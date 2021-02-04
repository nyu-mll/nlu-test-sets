# NLI Benchmark Analysis

## IRT Data

All model responses for NLI benchmarks are available at `nli-data` directory.

## Running IRT Analysis

(Optional) If you have additional dataset, prepare the model reponses, see more details [here](https://github.com/nyu-mll/nlu-test-sets), and add them to the `nli-data` folder. You should also add their metadata in the `task_metadata_nli.csv`

After that, run the following to estimate the IRT parameters:
```
bash irt_scripts/estimate_irt_params_nli.sh
```
You would need to adjust your environment path and the list of datasets if you have additional datasets. The parameters will then be stored in `params/nli-agreement` directory.

Now, you can run the `irt_scripts/plot_virt.ipynb` notebook to start IRT analysis. If you have additional datasets, you need to add them as well in the cell where we list all the datasets.

## Analyzing Agreement for SNLI and MNLI

You can use the `irt_scripts/preproc_scripts/generate_agreement_data.py` to breakdown MNLI and SNLI responses to agreement (3/5, 4/5, 5/5) responses.

