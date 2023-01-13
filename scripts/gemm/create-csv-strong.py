import sys

MEASUREMENTS_COUNT=11
lines = ["name,runtime,size,n_processors,nodes"]
dest="/cluster/home/bfrydrych/submissions/"
#job = sys.argv[1]
sets = ["STRONG_DATASET_1024", "STRONG_DATASET_2048", "STRONG_DATASET_4096", "STRONG_DATASET_8192"]
for set in sets:
    rootdir = f"/cluster/home/bfrydrych/submissions/{set}"
    cores = [1, 2, 4, 8, 16, 32]
    size = set.replace("STRONG_DATASET_", "")

    gemmBaseline = []
    for measurement in range(1, MEASUREMENTS_COUNT):
        with open(f"{rootdir}/gemm{measurement}.out", "r") as file0:
            gemmBaseline.append(file0.readlines()[-1])
    for num in gemmBaseline:
        for core in cores:
            lines.append(f"gemm-baseline,{num.strip()},{size},{core},1")

    #gemmBlas = []
    #for measurement in range(1, MEASUREMENTS_COUNT):
    #    with open(f"{rootdir}/gemm-blas{measurement}.out", "r") as fileBlas:
    #        gemmBlas.append(fileBlas.readlines()[-1])
    #for num in gemmBlas:
    #    for core in cores:
    #        lines.append(f"gemm-blas,{num.strip()},4096,{core},1")


    for core in cores:
        gemmMpiMeasurements = []
        gemmMpiSimpleMeasurements = []
        gemmOpenMpMeasurements = []
        for measurement in range(1, MEASUREMENTS_COUNT):
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

        for num in gemmMpiMeasurements:
            lines.append(f"gemm-mpi,{num.strip()},{size},{core},1")
        for num in gemmMpiSimpleMeasurements:
            lines.append(f"gemm-mpi-simple,{num.strip()},{size},{core},1")
        for num in gemmOpenMpMeasurements:
            lines.append(f"gemm-openmp,{num.strip()},{size},{core},1")




with open(f"{dest}/strong.csv", 'w') as f4:
    f4.write("\n".join(lines))

