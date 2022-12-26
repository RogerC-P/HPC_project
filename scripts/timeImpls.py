import os
import sys

from helper import get_files

def main():
    #dataset_sizes = [2**i for i in range(6, 12)]
    dataset_sizes = [2048]

    time_benches(dataset_sizes, "./linear-algebra/solvers/ludcmp")


def time_benches(dataset_sizes, *dirs):
    for dir in dirs:
        time_bench(dataset_sizes, dir)


def time_bench(dataset_sizes, dir):

    files = get_files(dir)
    base = files["base"]
    optimizations = files["opt"]

    for dataset_size in dataset_sizes:
        print(f"Running for dataset size {dataset_size}")

        print("{}".format(os.path.basename(base)), end =" ... ", flush=True)
        base_result = run_impl(base, dataset_size)
        print("{} seconds".format(base_result))

        for other_impl in optimizations:
            print("{}".format(os.path.basename(other_impl)), end =" ... ", flush=True)
            other_result = run_impl(other_impl, dataset_size)

            #percentage_improvement = round((other_result - base_result) / base_result * 100)
            speedup = round(base_result / other_result)
            msg = "{} seconds ({}x speedup)".format(other_result, speedup)

            if other_result < base_result:
                printGreen(msg)
            elif other_result > base_result:
                printRed(msg)
            else:
                print(msg)


def run_impl(impl:str, dataset_size, runs = 3):
    header = impl.replace(".c", "")

    if "mpi" in impl:
        compiler = "mpicc"
    else:
        compiler = "gcc"

    flags = ["-O3", "-std=c99", "-D_POSIX_C_SOURCE=200112L", "-O3", "-march=native"]
    if "fma" in impl:
        flags.append("-mfma")
    if "openmp" in impl:
        flags.append("-fopenmp")
    if "blas" in impl:
        flags.append("-mfma -fopenmp -I /usr/include/openblas -lopenblas")

    joined_flags = " ".join(flags)

    # Compile implementation
    if impl.endswith(".c"):
        os.system(f"{compiler} {joined_flags} -I utilities -I {header} utilities/polybench.c {impl} -DSIZE_DATASET={dataset_size} -DPOLYBENCH_TIME -o executable")
    
    outputs = []
    key = "OMP_NUM_THREADS"
    numc_cores = os.getenv(key, "X")
    np = 1
    # Run and get output
    for _ in range(runs):
        if "mpi" in impl:
            np = 2
            output = os.popen(f"mpirun -np {np} --oversubscribe ./executable 2>&1").read()
        else:
            output = os.popen("./executable 2>&1").read()
        results = {
            "name": os.path.basename(header),
            "runtime": float(output),
            "size": dataset_size,
            "n_processors": numc_cores,
            "nodes": np,
        }
        outputs.append(results)
    return outputs


def printRed(text): print("\033[91m{}\033[91m" .format(text))
def printGreen(text): print("\033[92m{}\033[92m" .format(text))
def printOrange(text): print("\033[35m{}\033[35m" .format(text), end=" ")
 

if __name__ == "__main__":
    main()
