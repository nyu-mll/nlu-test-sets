#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=mvirt
#SBATCH --output=sl_vi_en_1011.out
#SBATCH --error=sl_zh_en_1011.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --mem=30GB
#SBATCH --mail-user=pmh330@nyu.edu
#SBATCH --mail-type=END,FAIL

#module purge # unload all modules
#module load cuda/8.0.44
#module load cudnn/8.0v5.1

#conda activate /misc/vlgscratch4/BowmanGroup/pmh330/miniconda3/envs/env_mvirt
dim=$1
base_dir=$2
saved_param_dir=$3
file_name=$4

out_dir="/misc/vlgscratch4/BowmanGroup/pmh330/nlu-test-sets/${saved_param_dir}"

echo "starting" >> $out_dir/${file_name}/log_plot_stat.log 2>&1

/misc/vlgscratch4/BowmanGroup/pmh330/miniconda3/envs/env_mvirt/bin/python \
   irt_scripts/save_plot_stats.py \
   $dim $base_dir $saved_param_dir $file_name >> $out_dir/${file_name}/log_plot_stat.log 2>&1
