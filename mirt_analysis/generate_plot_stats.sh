BASE_DIR=$(pwd)

DISTS=('lognormal')
ALPH_STDS=( '0.20' '0.15' '0.25')
PARAM_STDS=('1.0')
DIMENSIONS=( 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 36 54 72 90 )

for dim in "${DIMENSIONS[@]}"
do
    sbatch $BASE_DIR/sb_plot.sbatch $dim
done
