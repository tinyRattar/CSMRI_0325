#!/bin/sh
#BSUB -q newgpu
#BSUB -o log/dOri_2_tr
#BSUB -J dOri_2_tr
#BSUB -R "select[ngpus >0] rusage[ngpus_excl_p=1]"

echo ======Job Start======
python main.py dOri_tr
