#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=shen_onlstm
#SBATCH --output=sl_vi_en_1011.out
#SBATCH --error=sl_zh_en_1011.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --mail-user=pmh330@nyu.edu
#SBATCH --mail-type=END,FAIL

#module purge # unload all modules
#module load cuda/8.0.44
#module load cudnn/8.0v5.1

conda activate env_mvirt
echo "starting" >> $OUT_DIR/log.log 2>&1

ALPHA_TRANS=identity
THETA_TRANS=identity
IN_DIR=$1
OUT_DIR=$2
seed=101
sample_size=1500
alpha_std=$3
item_std=$4
dim=$5
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
   --dimension $dim --verbose   >> $OUT_DIR/log.log 2>&1
