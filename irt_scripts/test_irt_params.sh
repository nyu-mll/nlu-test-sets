#BASE_DIR=/Users/phumon/Documents/Research/
# BASE_DIR=//Users/claravania/Projects
# BASE_DIR=/scratch/wh629/research/irt/irt_script
loss_type=$1

BASE_DIR=$(pwd)
echo "Base dir: $BASE_DIR"
SCRIPT_DIR=${BASE_DIR}/irt_scripts

IN_DIR=${BASE_DIR}/data
OUT_DIR=${BASE_DIR}/params_mvirt-${loss_type}
SEED=101

# adjust the following parameters according to chosen setup
# here we use our best parameters used in the paper.
# supported distributions: 'lognormal', 'beta', 'normal'
DISTS=('lognormal')
ALPH_STDS=( '0.20' '0.15' '0.25')
PARAM_STDS=('1.0')
LRS=( '0.01' '0.005' '0.001' '0.0005' )
# DIMENSIONS=( 90 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 36 54 72)
DIMENSIONS=( 1 2 3 15 36 54 72)
#  list of tasks to analyze
TASKS="boolq,cb,commonsenseqa,copa,cosmosqa,hellaswag,rte,snli,wic,qamr,arct,mcscript,mctaco,mutual,mutual-plus,quoref,socialiqa,squad-v2,wsc,mnli,mrqa-nq,newsqa,abductive-nli,arc-easy,arc-challenge,piqa,quail,winogrande"

# if want to use sampling instead of all examples
# not used if --no_subsample is specified
sample_size=1500
for lr in "${LRS[@]}"
do
for alpha_std in "${ALPH_STDS[@]}"
do
    for dim in "${DIMENSIONS[@]}"
    do
	for item_std in "${PARAM_STDS[@]}"
	do
            echo Alpha Std $alpha_std, Diff Guess Std $item_std
	    ALPHA_TRANS=identity
	    THETA_TRANS=identity
            sbatch $BASE_DIR/sb_test_wh.sbatch $IN_DIR $OUT_DIR $alpha_std $item_std $dim $loss_type $lr
            #sbatch $BASE_DIR/sb_test_wh.sbatch 
            #python \
	    #	$SCRIPT_DIR/variational_irt.py \
  	    #	--response_dir $IN_DIR \
  	    #	--out_dir $OUT_DIR \
  	    #	--seed $SEED \
  	    #	--discr 'lognormal' \
  	    #	--ability 'normal' \
  	    # 	--discr_transform $ALPHA_TRANS \
  	    # 	--ability_transform $THETA_TRANS \
            #   --datasets $TASKS \
            #   --sample_size $sample_size \
            #   --no_subsample \
            #   --alpha_std $alpha_std \
            #   --item_param_std $item_std \
            #   --lr $lr \
            #   --verbose
	done
    done
done
done
