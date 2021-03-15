#BASE_DIR=/Users/phumon/Documents/Research/
# BASE_DIR=//Users/claravania/Projects
# BASE_DIR=/scratch/wh629/research/irt/irt_script
BASE_DIR=$(pwd)
echo "Base dir: $BASE_DIR"
SCRIPT_DIR=${BASE_DIR}/irt_scripts

IN_DIR=${BASE_DIR}/data_synthetic
OUT_DIR=${BASE_DIR}/params_mvirt_sync
SEED=101

# adjust the following parameters according to chosen setup
# here we use our best parameters used in the paper.
# supported distributions: 'lognormal', 'beta', 'normal'
DISTS=('lognormal')
ALPH_STDS=( '0.4' '0.2' )
PARAM_STDS=('1.0' '2' '10')
DIMENSIONS=( 1 2 3 4 5 6 7 8 ) ## 13 14 15 16 17 18 36 54 72)
LR=(0.0001 0.001)
#  list of tasks to analyze
#TASKS="boolq,cb,commonsenseqa,copa,cosmosqa,hellaswag,rte,snli,wic,qamr,arct,mcscript,mctaco,mutual,mutual-plus,quoref,socialiqa,squad-v2,wsc,mnli,mrqa-nq,newsqa,abductive-nli,arc-easy,arc-challenge,piqa,quail,winogrande"
TASKS="sync_dim3_mean0_alpha-lognormal-4.00_theta-normal-4.00,sync_dim3_mean10_alpha-lognormal-2.00_theta-normal-2.00"

# if want to use sampling instead of all examples
# not used if --no_subsample is specified
sample_size=1500
for alpha_std in "${ALPH_STDS[@]}"
do
    for lr in "${LR[@]}"
    do
        for dim in "${DIMENSIONS[@]}"
        do
	    for item_std in "${PARAM_STDS[@]}"
	    do
                echo Alpha Std $alpha_std, Diff Guess Std $item_std
	        ALPHA_TRANS=identity
	        THETA_TRANS=identity
                sbatch $BASE_DIR/sb_test.sbatch $IN_DIR $OUT_DIR $alpha_std $item_std $dim $lr 
	    done
        done
    done
done
