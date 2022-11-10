import os


def main():
    time_benches("./linear-algebra/solvers/ludcmp")


def time_benches(*dirs):
    for dir in dirs:
        time_bench(dir)


def time_bench(dir):
    dir = os.path.normpath(dir)
    bench_name = os.path.basename(dir);

    print("Checking performance of bench '{}'".format(bench_name))

    dataset_sizes = ["MINI_DATASET", "SMALL_DATASET", "MEDIUM_DATASET", "LARGE_DATASET", "EXTRALARGE_DATASET"]
    dataset_sizes = ["LARGE_DATASET"]

    for dataset_size in dataset_sizes:
        print(f"Running for dataset size {dataset_size}")

        # Assume base benchmark has same name as containing directory
        base_name = bench_name + ".c"
        base_impl = dir + "/" + base_name
        other_impls = map(lambda name: dir + "/" + name, filter(lambda name: name.endswith(".c") and name != base_name, os.listdir(dir)))

        print("{}".format(os.path.basename(base_impl)), end =" ... ", flush=True)
        base_result = run_impl(base_impl, dataset_size)
        print("{} cycles".format(base_result))

        for other_impl in other_impls:
            print("{}".format(os.path.basename(other_impl)), end =" ... ", flush=True)
            other_result = run_impl(other_impl, dataset_size)

            percentage_improvement = round((other_result - base_result) / base_result * 100)
            msg = "{} cycles ({:+}%)".format(other_result, percentage_improvement);

            if other_result < base_result:
                printGreen(msg)
            elif other_result > base_result:
                printRed(msg)
            else:
                print(msg)



def run_impl(impl, dataset_size = "MEDIUM_DATASET"):
    header = impl.replace(".c", "")

    if "mpi" in impl:
        compiler = "mpicc"
    else:
        compiler = "gcc"

    flags = ["-O3"]
    if "fma" in impl:
        flags.append("-mfma")
    if "openmp" in impl:
        flags.append("-fopenmp")

    joined_flags = " ".join(flags)

    # Compile implementation
    os.system(f"{compiler} {joined_flags} -I utilities -I {header} utilities/polybench.c {impl} -D{dataset_size} -DPOLYBENCH_TIME -DPOLYBENCH_CYCLE_ACCURATE_TIMER -o executable")
    # Run and get output
    output = os.popen("./executable 2>&1").read()
    time = int(output)
    return time


def printRed(text): print("\033[91m{}\033[00m" .format(text))
def printGreen(text): print("\033[92m{}\033[00m" .format(text))
 

if __name__ == "__main__":
    main()