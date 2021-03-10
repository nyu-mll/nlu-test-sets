#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=mvirt
#SBATCH --output=sl_vi_en_1011.out
#SBATCH --error=sl_zh_en_1011.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --mail-user=pmh330@nyu.edu
#SBATCH --mail-type=END,FAIL

#module purge # unload all modules
#module load cuda/8.0.44
#module load cudnn/8.0v5.1

#conda activate /misc/vlgscratch4/BowmanGroup/pmh330/miniconda3/envs/env_mvirt
#TASKS="boolq,cb,commonsenseqa,copa,cosmosqa,hellaswag,rte,snli,wic,qamr,arct,mcscript,mctaco,mutual,mutual-plus,quoref,socialiqa,squad-v2,wsc,mnli,mrqa-nq,newsqa,abductive-nli,arc-easy,arc-challenge,piqa,quail,winogrande"
#TASKS='snli'
TASKS="sync_dim3_alpha-lognormal-2.00_theta-normal-2.00,sync_dim3_alpha-lognormal-7.00_theta-normal-10.00"

ALPHA_TRANS=identity
THETA_TRANS=identity
IN_DIR=$1
OUT_DIR=$2
SEED=101
sample_size=1500
alpha_std=$3
item_std=$4
dim=$5
lr=$6
steps=4000

logging_dir="lr-${lr}-steps-${steps}-alpha-${alpha_std}-${ALPHA_TRANS}-dim${dim}_theta-${normal}-${THETA_TRANS}_nosubsample_${item_std}_${alpha_std}"

mkdir -p $OUT_DIR/$logging_dir
echo "starting" >> $OUT_DIR/$logging_dir/log.log 2>&1

echo "$IN_DIR outdir: $OUT_DIR $alpha_std $item_std $dim"
/misc/vlgscratch4/BowmanGroup/pmh330/miniconda3/envs/env_mvirt/bin/python \
    $(pwd)/irt_scripts/multi_virt_v2.py \
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
   --lr $lr --steps $steps \
   --num_particles 8 \
   --dimension $dim --verbose   >> $OUT_DIR/$logging_dir/log.log 2>&1
