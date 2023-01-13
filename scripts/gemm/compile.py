import os

<<<<<<< HEAD
problemSizes = ["DATASET_5000","DATASET_6300","DATASET_7938","DATASET_10000","DATASET_12600","DATASET_15874","DATASET_1024","DATASET_2048","DATASET_4096","DATASET_8192"]
#problemSizes = ["DATASET_12600"]
rootDir = "~/projects/PolyBenchC-4.2.1/"
mklDir = os.environ['MKLROOT']


def compileGemmBaseline(problemSizes):
    compile("gcc", "gemm.c", "gemm", "", "", problemSizes)

def compileGemmBlas(problemSizes):
    compile("gcc", "gemm-blas.c", "gemm-blas", f"-L{mklDir}/lib/intel64 -I {mklDir}/include", f"-Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl  -DMKL_ILP64  -m64 -I {mklDir}/include", problemSizes)

def compileGemmMpi(problemSizes):
    compile("mpicc", "gemm-mpi.c", "gemm-mpi", "", "-mavx -march=native -mfma", problemSizes)

def compileGemmMpiSimple(problemSizes):
    compile("mpicc", "gemm-mpi-simple.c", "gemm-mpi-simple", "", "", problemSizes)

def compileGemmOpenMp(problemSizes):
    compile("gcc", "gemm-openmp.c", "gemm-openmp", "", "-fopenmp -mavx -march=native -mfma", problemSizes)

def compile(compiler, fileToCompileName, execName, includes, compileFlags, problemSizes):
    print(f"Compiling {fileToCompileName}")
    impl = f"{rootDir}/linear-algebra/blas/gemm/{fileToCompileName}"  
    for problemSize in problemSizes:
        command = f"{compiler} -O3 -I {rootDir}/utilities -I {rootDir}/linear-algebra/blas/gemm/customdatasizes {includes} {impl} {rootDir}/utilities/polybench.c {compileFlags} -D{problemSize} -DPOLYBENCH_TIME -o {rootDir}/linear-algebra/blas/gemm/exec/{execName}-{problemSize}"        
        os.system(command)


=======
problemSizes = ["MINI_DATASET","SMALL_DATASET","MEDIUM_DATASET","LARGE_DATASET","EXTRALARGE_DATASET",]
rootDir = "~/projects/PolyBenchC-4.2.1/"

def compileGemmBaseline(problemSizes):
    compile("gcc", "gemm.c", "gemm", "", problemSizes)

def compileGemmBlas(problemSizes):
    compile("gcc", "gemm-blas.c", "gemm-blas", "-lopenblas", problemSizes)

def compileGemmMpi(problemSizes):
    compile("mpicc", "gemm-mpi.c", "gemm-mpi", "-mavx -march=native -mfma", problemSizes)

def compileGemmMpiSimple(problemSizes):
    compile("mpicc", "gemm-mpi-simple.c", "gemm-mpi-simple", "", problemSizes)

def compileGemmOpenMp(problemSizes):
    compile("gcc", "gemm-openmp.c", "gemm-openmp", "-fopenmp -mavx -march=native -mfma", problemSizes)

def compile(compiler, fileToCompileName, execName, compileFlags, problemSizes):
    impl = f"{rootDir}/linear-algebra/blas/gemm/{fileToCompileName}"    
    for problemSize in problemSizes:
        os.system(f"{compiler} -O3 -I {rootDir}/utilities -I {rootDir}/linear-algebra/blas/gemm {impl} {rootDir}/utilities/polybench.c {compileFlags} -D{problemSize} -DPOLYBENCH_TIME -o {rootDir}/linear-algebra/blas/gemm/exec/{execName}-{problemSize}")
>>>>>>> 1e718dd40034efc7b14101b2267627c5c54531d0


compileGemmBaseline(problemSizes)
compileGemmBlas(problemSizes)
compileGemmMpi(problemSizes)
compileGemmMpiSimple(problemSizes)
<<<<<<< HEAD
compileGemmOpenMp(problemSizes)
=======
compileGemmOpenMp(problemSizes)
>>>>>>> 1e718dd40034efc7b14101b2267627c5c54531d0
