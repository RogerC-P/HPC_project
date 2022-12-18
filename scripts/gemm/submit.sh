#!/bin/bash
GEMM_EXECS=~/gemm-execs/
SUBDIR=~/submissions/${1}
DIR=~/submissions/${1}/jobs/
mkdir -p ${DIR}

module load gcc/8.2.0
for ((i=1;i<=10;i++)); 
do
   sbatch --output ${SUBDIR}/gemm${i}.out --wrap "lscpu;~/gemm-execs/gemm" > ${DIR}/gemm${i}.out
done


module load openmpi/4.1.4
for ((core=0;core<=5;core++));
do
    CORES=$((2**$core))
    CORESDIR=${SUBDIR}/${CORES}
    mkdir -p ${CORESDIR}
    mkdir -p ${CORESDIR}/jobs

    export OMP_NUM_THREADS=${CORES}
    for ((i=1;i<=10;i++));
    do
        sbatch --output ${CORESDIR}/gemm-openmp${i}.out --cpus-per-task ${CORES} --wrap "lscpu;~/gemm-execs/gemm-openmp" > ${CORESDIR}/jobs/gemm-openmp${i}.out
        sbatch --output ${CORESDIR}/gemm-mpi${i}.out --ntasks ${CORES} --wrap "lscpu;mpirun ~/gemm-execs/gemm-mpi" > ${CORESDIR}/jobs/gemm-mpi${i}.out
        sbatch --output ${CORESDIR}/gemm-mpi-simple${i}.out --ntasks ${CORES} --wrap "lscpu;mpirun ~/gemm-execs/gemm-mpi-simple" > ${CORESDIR}/jobs/gemm-mpi-simple${i}.out
    done

    
done


CORES=48
CORESDIR=${SUBDIR}/${CORES}
mkdir -p ${CORESDIR}
mkdir -p ${CORESDIR}/jobs
export OMP_NUM_THREADS=${CORES}
for ((i=1;i<=10;i++));
do
    sbatch --output ${CORESDIR}/gemm-openmp${i}.out --cpus-per-task ${CORES} --wrap "lscpu;~/gemm-execs/gemm-openmp" > ${CORESDIR}/jobs/gemm-openmp${i}.out
    sbatch --output ${CORESDIR}/gemm-mpi${i}.out --ntasks ${CORES} --wrap "lscpu;mpirun ~/gemm-execs/gemm-mpi" > ${CORESDIR}/jobs/gemm-mpi${i}.out
    sbatch --output ${CORESDIR}/gemm-mpi-simple${i}.out --ntasks ${CORES} --wrap "lscpu;mpirun ~/gemm-execs/gemm-mpi-simple" > ${CORESDIR}/jobs/gemm-mpi-simple${i}.out
done
