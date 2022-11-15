CC=mpicc

override CFLAGS += -std=c99 -D_POSIX_C_SOURCE=200112L -O3 -march=native -fopenmp \
									 -DPOLYBENCH_TIME -DN_RUNS=10

# override CFLAGS += -g -march=native -DMINI_DATASET -DPOLYBENCH_DUMP_ARRAYS

SHARED = $(wildcard shared/*)
GEMM = $(wildcard linear-algebra/blas/gemm/*)
LU = $(wildcard linear-algebra/solvers/lu/*)
LUDCMP = $(wildcard linear-algebra/solvers/ludcmp/*)

polybench.o: utilities/polybench.c utilities/polybench.h
	$(CC) $(CFLAGS) -c -I utilities utilities/polybench.c -o $@ 

gemm.o: $(SHARED) $(GEMM)
	$(CC) $(CFLAGS) -c -I utilities -I shared -I linear-algebra/blas/gemm linear-algebra/blas/gemm/gemm.c -o $@

lu.o: $(SHARED) $(LU)
	$(CC) $(CFLAGS) $(PB_FLAGS) -c -I utilities -I shared -I linear-algebra/solvers/lu linear-algebra/solvers/lu/lu.c -o $@

ludcmp.o: $(SHARED) $(LUDCMP)
	$(CC) $(CFLAGS) -c -I utilities -I shared -I linear-algebra/solvers/ludcmp linear-algebra/solvers/ludcmp/ludcmp.c -o $@

gemm: polybench.o gemm.o
	$(CC) $(CFLAGS) polybench.o gemm.o -o gemm

lu: polybench.o lu.o
	$(CC) $(CFLAGS) polybench.o lu.o -o lu

ludcmp: polybench.o ludcmp.o
	$(CC) $(CFLAGS) polybench.o ludcmp.o -o ludcmp

N = 1
M = 1
T = 1

CPU = EPYC_7763

openmp_job: $(benchmark)
	mkdir -p results
	export OMP_NUM_THREADS=$(T); sbatch --output="results/$(benchmark)-$(T)" --open-mode=truncate \
				--ntasks=1 --cpus-per-task=$(T) \
				--constraint=$(CPU) --time=1:00 \
				--wrap="./$(benchmark)"

mpi_cmd = mpirun -n $(M) --map-by node:PE=$(T) $(benchmark)

mpi_job: $(benchmark)
	mkdir -p results
	export OMP_NUM_THREADS=$(T); sbatch --output="results/$(benchmark)-$(N)-$(M)-$(T)" --open-mode=truncate \
				--ntasks=$(N) --ntasks-per-node=$(T) \
				--constraint=$(CPU) \
				--wrap="unset LSB_AFFINITY_HOSTFILE; $(mpi_cmd)"

clean:
	rm -f *.o gemm lu ludcmp
