import sys

job = sys.argv[1]
rootdir = f"/cluster/home/bfrydrych/submissions/{job}"
cores = [1, 2, 4, 8, 16, 32, 48]
for core in cores:
    gemmMpiMeasurements = []
    gemmMpiSimpleMeasurements = []
    gemmOpenMpMeasurements = []
    for measurement in range(1, 10):
        with open(f"{rootdir}/{core}/gemm-mpi{measurement}.out", "r") as file1:
            gemmMpiMeasurements.append(file1.readlines()[-1])
        with open(f"{rootdir}/{core}/gemm-mpi-simple{measurement}.out", "r") as file2:
            gemmMpiSimpleMeasurements.append(file2.readlines()[-1])
        with open(f"{rootdir}/{core}/gemm-openmp{measurement}.out", "r") as file3:
            gemmOpenMpMeasurements.append(file3.readlines()[-1])

    with open(f"{rootdir}/{core}/results-gemm-mpi.out", 'w') as f1:
        for measure in gemmMpiMeasurements:
            f1.write(f"{measure}")
    with open(f"{rootdir}/{core}/results-gemm-mpi-simple.out", 'w') as f2:
        for measure in gemmMpiSimpleMeasurements:
            f2.write(f"{measure}")
    with open(f"{rootdir}/{core}/results-gemm-openmp.out", 'w') as f3:
        for measure in gemmOpenMpMeasurements:
            f3.write(f"{measure}")

#lines = ["name,runtime,size,n_processors,nodes"]
