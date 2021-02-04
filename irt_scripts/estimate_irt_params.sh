# BASE_DIR=/Users/phumon/Documents/Research/
BASE_DIR=//Users/claravania/Projects
# BASE_DIR=/scratch/wh629/research/irt/irt_script
SCRIPT_DIR=${BASE_DIR}/nlu-test-sets/irt_scripts

IN_DIR=${BASE_DIR}/nlu-test-sets/data
OUT_DIR=${BASE_DIR}/nlu-test-sets/params_anli
SEED=101

# adjust the following parameters according to chosen setup
# here we use our best parameters used in the paper.
# supported distributions: 'lognormal', 'beta', 'normal'
DISTS=('lognormal')
ALPH_STDS=('0.30')
PARAM_STDS=('1.0')

#  list of tasks to analyze
TASKS="boolq,cb,commonsenseqa,copa,cosmosqa,hellaswag,rte,snli,wic,qamr,arct,mcscript,mctaco,mutual,mutual-plus,quoref,socialiqa,squad-v2,wsc,mnli,mrqa-nq,newsqa,abductive-nli,arc-easy,arc-challenge,piqa,quail,winogrande,adversarial-nli"

# if want to use sampling instead of all examples
# not used if --no_subsample is specified
sample_size=1500
for alpha_std in "${ALPH_STDS[@]}"
do
	for item_std in "${PARAM_STDS[@]}"
	do
		echo Alpha Std $alpha_std, Diff Guess Std $item_std
		ALPHA_TRANS=identity
		THETA_TRANS=identity
    python \
			$SCRIPT_DIR/variational_irt.py \
  			--response_dir $IN_DIR \
  			--out_dir $OUT_DIR \
  			--seed $SEED \
  			--discr 'lognormal' \
  			--ability 'normal' \
  			--discr_transform $ALPHA_TRANS \
  			--ability_transform $THETA_TRANS \
        --datasets $TASKS \
        --sample_size $sample_size \
        --no_subsample \
        --alpha_std $alpha_std \
        --item_param_std $item_std \
  			--verbose
	done
done
