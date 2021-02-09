#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=mvirt
#SBATCH --output=sl_vi_en_1011.out
#SBATCH --error=sl_zh_en_1011.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --mail-user=wh629@nyu.edu
#SBATCH --mail-type=END,FAIL

#module purge # unload all modules
#module load cuda/8.0.44
#module load cudnn/8.0v5.1

#source activate /scratch/wh629/research/irt/nlu-test-sets/env
TASKS="boolq,cb,commonsenseqa,copa,cosmosqa,hellaswag,rte,snli,wic,qamr,arct,mcscript,mctaco,mutual,mutual-plus,quoref,socialiqa,squad-v2,wsc,mnli,mrqa-nq,newsqa,abductive-nli,arc-easy,arc-challenge,piqa,quail,winogrande"

ALPHA_TRANS=identity
THETA_TRANS=identity
IN_DIR=$1
OUT_DIR=$2
SEED=101
sample_size=1500
alpha_std=$3
item_std=$4
dim=$5
loss_type=$6

logging_dir="alpha-${alpha_std}-${ALPHA_TRANS}-dim${dim}_theta-${normal}-${THETA_TRANS}_nosubsample_${item_std}_${alpha_std}"

mkdir -p $OUT_DIR/$logging_dir
echo "starting" >> $OUT_DIR/$logging_dir/log.log 2>&1

echo "$IN_DIR outdir: $OUT_DIR $alpha_std $item_std $dim"
/scratch/wh629/research/irt/nlu-test-sets/env/bin/python \
    $(pwd)/mirt_analysis/multi_virt_v2.py \
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
   --loss_type $loss_type \
   --dimension $dim --verbose   >> $OUT_DIR/$logging_dir/log.log 2>&1
