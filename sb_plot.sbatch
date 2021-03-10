#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=mvirt
#SBATCH --output=sl_vi_en_1011.out
#SBATCH --error=sl_zh_en_1011.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --mail-user=pmh330@nyu.edu
#SBATCH --mail-type=END,FAIL

#module purge # unload all modules
#module load cuda/8.0.44
#module load cudnn/8.0v5.1

#conda activate /misc/vlgscratch4/BowmanGroup/pmh330/miniconda3/envs/env_mvirt
dim=$1
base_dir=$(pwd)
out_dir="/misc/vlgscratch4/BowmanGroup/pmh330/nlu-test-sets/params_mvirt"
logging_dir="alpha-lognormal-identity-dim${dim}_theta-normal-identity_nosubsample_1.00_0.15"

echo "starting" >> $out_dir/$logging_dir/log_plot_stat.log 2>&1

/misc/vlgscratch4/BowmanGroup/pmh330/miniconda3/envs/env_mvirt/bin/python \
   mirt_analysis/save_plot_stats.py \
   $dim $base_dir  >> $out_dir/$logging_dir/log_plot_stat.log 2>&1
