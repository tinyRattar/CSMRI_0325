#!/bin/sh
#BSUB -q newgpu
#BSUB -o log/sr15f_dOri_t2
#BSUB -J sr15f_dOri_t2
#BSUB -R "select[ngpus >0] rusage[ngpus_excl_p=1]"

echo ======Job Start======
python main.py sr15f_dOri_t2
