#!/bin/bash
CPU='EPYC_7763'
GEMM_EXECS=~/gemm-execs/
SUBDIR=~/submissions/${1}
DIR=~/submissions/${1}/jobs/

SET=${2}
MEM="1G"
if [[ "$SET" == 'DATASET_15874' || "$SET" == 'DATASET_12600' ]]; then
    MEM="6G"
fi
if [[ "$SET" == 'DATASET_8192' || "$SET" == 'DATASET_10000' || "$SET" == 'DATASET_7938' ]]; then
    MEM="4G"
fi
if [[ "$SET" == 'DATASET_6300' || "$SET" == 'DATASET_5000' || "$SET" == 'DATASET_4096' ]]; then
    MEM="2G"
fi



mkdir -p ${DIR}


module load gcc/8.2.0
for ((i=1;i<=10;i++)); 
do
   sbatch --output ${SUBDIR}/gemm${i}.out --constraint=${CPU} --mem-per-cpu=${MEM} --wrap "lscpu;~/gemm-execs/gemm-${SET}" > ${DIR}/gemm${i}.out
done

module load openblas/0.3.20
module load openmpi/4.1.4
module load intel/2020.0
for ((core=0;core<=5;core++));
do
    CORES=$((2**$core))
    CORESDIR=${SUBDIR}/${CORES}
    mkdir -p ${CORESDIR}
    mkdir -p ${CORESDIR}/jobs

    export OMP_NUM_THREADS=${CORES}
    export OPENBLAS_NUM_THREADS=${CORES}
    for ((i=1;i<=10;i++));
    do
        #sbatch --output ${CORESDIR}/gemm-blas${i}.out --cpus-per-task ${CORES} --constraint=${CPU} --mem-per-cpu=${MEM} --wrap "lscpu;~/gemm-execs/gemm-blas-${SET}" > ${CORESDIR}/gemm-blas${i}.out
        sbatch --output ${CORESDIR}/gemm-openmp${i}.out --cpus-per-task ${CORES} --constraint=${CPU} --mem-per-cpu=${MEM} --wrap "lscpu;~/gemm-execs/gemm-openmp-${SET}" > ${CORESDIR}/jobs/gemm-openmp${i}.out
        sbatch --output ${CORESDIR}/gemm-mpi${i}.out --ntasks ${CORES} --constraint=${CPU} --mem-per-cpu=${MEM} --wrap "lscpu;mpirun ~/gemm-execs/gemm-mpi-${SET}" > ${CORESDIR}/jobs/gemm-mpi${i}.out
        sbatch --output ${CORESDIR}/gemm-mpi-simple${i}.out --ntasks ${CORES} --constraint=${CPU} --mem-per-cpu=${MEM} --wrap "lscpu;mpirun ~/gemm-execs/gemm-mpi-simple-${SET}" > ${CORESDIR}/jobs/gemm-mpi-simple${i}.out
    done

    
done
