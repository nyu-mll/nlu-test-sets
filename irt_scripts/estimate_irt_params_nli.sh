BASE_DIR=//Users/claravania/Projects
SCRIPT_DIR=${BASE_DIR}/nlu-test-sets/irt_scripts

IN_DIR=${BASE_DIR}/nlu-test-sets/nli-data
OUT_DIR=${BASE_DIR}/nlu-test-sets/params/nli-agreement
SEED=101

# adjust the following parameters according to chosen setup
# here we use our best parameters used in the paper.
# supported distributions: 'lognormal', 'beta', 'normal'
DISTS=('lognormal')
ALPH_STDS=('0.30')
PARAM_STDS=('1.0')

mkdir -p params/nli-agreement

#  list of tasks to analyze
TASKS="adversarial_nli_r1,adversarial_nli_r2,adversarial_nli_r3,mnli3,mnli4,mnli5,mnli_mismatched3,mnli_mismatched4,mnli_mismatched5,snli3,snli4,snli5,ni-base,ni-contrast,ni-edit-other,ni-edit-premise,ni-paragraph,san-base-news,san-base-wiki,san-sim-news,san-sim-wiki,san-translate-wiki"

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
