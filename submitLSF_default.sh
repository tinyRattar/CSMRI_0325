#!/bin/sh
#BSUB -q newgpu
#BSUB -o log/15f_cv8
#BSUB -J cv8
#BSUB -R "select[ngpus >0] rusage[ngpus_excl_p=1]"

echo ======Job Start======
python main.py cv8
