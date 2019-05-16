#!/bin/sh
#BSUB -q newgpu
#BSUB -o log/30_c10_t2
#BSUB -J 30_c10_t2
#BSUB -R "select[ngpus >0] rusage[ngpus_excl_p=1]"

echo ======Job Start======
python main.py sr30f_c10_t2
