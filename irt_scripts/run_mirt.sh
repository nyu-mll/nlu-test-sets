#BASE_DIR=/Users/phumon/Documents/Research/
# BASE_DIR=//Users/claravania/Projects
# BASE_DIR=/scratch/wh629/research/irt/irt_script
BASE_DIR=$(pwd)
echo "Base dir: $BASE_DIR"
SCRIPT_DIR=${BASE_DIR}/irt_scripts

IN_DIR=${BASE_DIR}/data_sync/will_diff_skills
OUT_DIR=${BASE_DIR}/params_posterior_varied_will_diff_skills
SEED=101

mkdir -p ${OUT_DIR}

# adjust the following parameters according to chosen setup
# here we use our best parameters used in the paper.
# supported distributions: 'lognormal', 'beta', 'normal'
DISTS=('lognormal') # 'normal')
ALPH_STDS=( '0.5' ) # '0.6' )
PARAM_STDS=('1' '5')
DIMENSIONS=( 1 2 3 4 5 6 7 8 ) ## 13 14 15 16 17 18 36 54 72)
LR=(0.0001)
#  list of tasks to analyze

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
                for dist in "${DISTS[@]}"
                do
                    echo Alpha Std $alpha_std, Diff Guess Std $item_std
	            ALPHA_TRANS=positive
	            THETA_TRANS=identity
                    sbatch $BASE_DIR/sb_run_mirt.sbatch $IN_DIR $OUT_DIR $alpha_std $item_std $dim $lr $dist
                done
	    done
        done
    done
done
