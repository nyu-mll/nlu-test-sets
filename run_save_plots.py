import os
import glob
import sys


base_dir="/misc/vlgscratch4/BowmanGroup/pmh330/nlu-test-sets"
save_dir="params_mvirt_sync"
output_path="/misc/vlgscratch4/BowmanGroup/pmh330/nlu-test-sets/params_mvirt_sync/lr-0.0001-steps-$5500-alpha-lognormal-identity-dim*_theta-normal-identity_nosubsample_*/params.p"
for file_name in glob.glob(output_path):
    names = file_name.split('/')[-2].replace('$', '\$')
    dim = file_name.split("dim")[1].split("_")[0]
    print("dim: ", dim)
    print("names: ", names)
    os.system("sbatch sb_plot.sbatch {} {} {} {}".format(dim, base_dir, save_dir, names)) 

