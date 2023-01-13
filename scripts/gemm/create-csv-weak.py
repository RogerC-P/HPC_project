import sys

MEASUREMENTS_COUNT=11
lines = ["name,runtime,size,n_processors,nodes"]
dest="/cluster/home/bfrydrych/submissions/"
#job = sys.argv[1]
sets = {"DATASET_5000": 1, "DATASET_6300": 2, "DATASET_7938": 4, "DATASET_10000": 8, "DATASET_12600": 16, "DATASET_15874": 32}
for set in sets:
    rootdir = f"/cluster/home/bfrydrych/submissions/WEAK/{set}"
    
    size = set.replace("DATASET_", "")
    
    #gemmBlas = []
    #for measurement in range(1, MEASUREMENTS_COUNT):
    #    with open(f"{rootdir}/gemm-blas{measurement}.out", "r") as fileBlas:
    #        gemmBlas.append(fileBlas.readlines()[-1])
    #for num in gemmBlas:
    #    for core in cores:
    #        lines.append(f"gemm-blas,{num.strip()},4096,{core},1")


    core = sets[set]
    gemmMpiMeasurements = []
    gemmMpiSimpleMeasurements = []
    gemmOpenMpMeasurements = []
    gemmBaseline = []
    for measurement in range(1, MEASUREMENTS_COUNT):
        with open(f"{rootdir}/{core}/gemm{measurement}.out", "r") as file0:
            gemmBaseline.append(file0.readlines()[-1])
        with open(f"{rootdir}/{core}/gemm-mpi{measurement}.out", "r") as file1:
            gemmMpiMeasurements.append(file1.readlines()[-1])
        with open(f"{rootdir}/{core}/gemm-mpi-simple{measurement}.out", "r") as file2:
            gemmMpiSimpleMeasurements.append(file2.readlines()[-1])
        with open(f"{rootdir}/{core}/gemm-openmp{measurement}.out", "r") as file3:
            gemmOpenMpMeasurements.append(file3.readlines()[-1])

    with open(f"{rootdir}/{core}/results-gemm-baseline.out", 'w') as f0:
        for measure in gemmBaseline:
            f0.write(f"{measure}")
    with open(f"{rootdir}/{core}/results-gemm-mpi.out", 'w') as f1:
        for measure in gemmMpiMeasurements:
            f1.write(f"{measure}")
    with open(f"{rootdir}/{core}/results-gemm-mpi-simple.out", 'w') as f2:
        for measure in gemmMpiSimpleMeasurements:
            f2.write(f"{measure}")
    with open(f"{rootdir}/{core}/results-gemm-openmp.out", 'w') as f3:
        for measure in gemmOpenMpMeasurements:
            f3.write(f"{measure}")
    
    for num in gemmBaseline:
        lines.append(f"gemm-baseline,{num.strip()},{size},{core},1")
    for num in gemmMpiMeasurements:
        lines.append(f"gemm-mpi,{num.strip()},{size},{core},1")
    for num in gemmMpiSimpleMeasurements:
        lines.append(f"gemm-mpi-simple,{num.strip()},{size},{core},1")
    for num in gemmOpenMpMeasurements:
        lines.append(f"gemm-openmp,{num.strip()},{size},{core},1")


with open(f"{dest}/weak.csv", 'w') as f4:
    f4.write("\n".join(lines))

