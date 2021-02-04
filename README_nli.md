# NLI Benchmark Analysis

## IRT Data

All model responses for NLI benchmarks are available at `nli-data` directory.

For [NLI with interventions](https://github.com/google-research-datasets/Textual-Entailment-New-Protocols) `ni-xxx` and [semi automatic NLI](https://github.com/nyu-mll/semi-automatic-nli) `san-xxx`, example index follows the full response file, i.e., `nli_intervention_irt_all_coded.csv` and `semi_auto_nli_irt_all_coded`. Both tasks are implemented in [jiant](https://github.com/nyu-mll/jiant/tree/irt-nli), and each task's test set combines all examples. For NLI intervention, the (development) examples are ordered based on `base, contrast, edit-other, edit-premise, paragraph`. For semi automatic NLI, the (test) examples are ordered based on `base-news, base-wiki, sim-news, sim-wiki, translate-wiki`.


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




