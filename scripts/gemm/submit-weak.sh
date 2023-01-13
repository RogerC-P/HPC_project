#!/bin/bash

SET=${2}
CORES=${3}
CPU='EPYC_7763'
GEMM_EXECS=~/gemm-execs/
SUBDIR=~/submissions/${1}/${SET}
DIR=~/submissions/${1}/${SET}/jobs/




MEM="1G"
MEM_GEMM="1G"
TIME_GEMM="1:00:00"
if [[ "$SET" == 'DATASET_15874' || "$SET" == 'DATASET_12600' ]]; then
    MEM="4G"
    MEM_GEMM="8G"
    TIME_GEMM="4:00:00"
fi
if [[ "$SET" == 'DATASET_8192' || "$SET" == 'DATASET_10000' || "$SET" == 'DATASET_7938' ]]; then
    MEM="4G"
    MEM_GEMM="4G"
fi
if [[ "$SET" == 'DATASET_6300' || "$SET" == 'DATASET_5000' || "$SET" == 'DATASET_4096' ]]; then
    MEM="2G"
    MEM_GEMM="2G"
fi



mkdir -p ${DIR}


module load gcc/8.2.0



module load openblas/0.3.20
module load openmpi/4.1.4

    
CORESDIR=${SUBDIR}/${CORES}
mkdir -p ${CORESDIR}/jobs

export OMP_NUM_THREADS=${CORES}
export OPENBLAS_NUM_THREADS=${CORES}
for ((i=1;i<=10;i++));
do
    sbatch --output ${CORESDIR}/gemm${i}.out --constraint=${CPU} --mem-per-cpu=${MEM_GEMM} --time=${TIME_GEMM} --wrap "lscpu;~/gemm-execs/gemm-${SET}" > ${CORESDIR}/jobs/gemm${i}.out
    #sbatch --output ${CORESDIR}/gemm-blas${i}.out --cpus-per-task ${CORES} --constraint=${CPU} --mem-per-cpu=${MEM} --wrap "lscpu;~/gemm-execs/gemm-blas-${SET}" > ${CORESDIR}/gemm-blas${i}.out
    sbatch --output ${CORESDIR}/gemm-openmp${i}.out --cpus-per-task ${CORES} --constraint=${CPU} --mem-per-cpu=${MEM} --wrap "lscpu;~/gemm-execs/gemm-openmp-${SET}" > ${CORESDIR}/jobs/gemm-openmp${i}.out
    sbatch --output ${CORESDIR}/gemm-mpi${i}.out --ntasks ${CORES} --constraint=${CPU} --mem-per-cpu=${MEM} --wrap "lscpu;mpirun ~/gemm-execs/gemm-mpi-${SET}" > ${CORESDIR}/jobs/gemm-mpi${i}.out
    sbatch --output ${CORESDIR}/gemm-mpi-simple${i}.out --ntasks ${CORES} --constraint=${CPU} --mem-per-cpu=${MEM} --wrap "lscpu;mpirun ~/gemm-execs/gemm-mpi-simple-${SET}" > ${CORESDIR}/jobs/gemm-mpi-simple${i}.out
done

    

