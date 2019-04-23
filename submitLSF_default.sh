#!/bin/sh
#BSUB -q newgpu
#BSUB -o log/3d_dOri_t4_l8
#BSUB -J 3d_dOrit4_l8
#BSUB -R "select[ngpus >0] rusage[ngpus_excl_p=2]"

echo ======Job Start======
python main.py default2
