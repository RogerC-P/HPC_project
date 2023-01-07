import os

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


compileGemmBaseline(problemSizes)
compileGemmBlas(problemSizes)
compileGemmMpi(problemSizes)
compileGemmMpiSimple(problemSizes)
compileGemmOpenMp(problemSizes)
