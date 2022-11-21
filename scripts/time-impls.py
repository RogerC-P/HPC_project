import os
import sys

def main():
    time_benches("./linear-algebra/blas/gemm", "./linear-algebra/solvers/ludcmp")


def time_benches(*dirs):
    for dir in dirs:
        time_bench(dir)


def time_bench(dir):
    dir = os.path.normpath(dir)
    bench_name = os.path.basename(dir);

    print("Checking performance of bench '{}'".format(bench_name))

    # Assume base benchmark has same name as containing directory
    base_name = bench_name + ".c"
    base_impl = dir + "/" + base_name
    other_impls = map(lambda name: dir + "/" + name, filter(lambda name: name.endswith(".c") and name != base_name, os.listdir(dir)))

    print("{}".format(os.path.basename(base_impl)), end =" ... ", flush=True)
    base_result = run_impl(base_impl)
    print("{}".format(base_result))

    for other_impl in other_impls:
        print("{}".format(os.path.basename(other_impl)), end =" ... ", flush=True)
        other_result = run_impl(other_impl)

        percentage_improvement = round((other_result - base_result) / base_result * 100)
        msg = "{} ({:+}%)".format(other_result, percentage_improvement);

        if other_result < base_result:
            printGreen(msg)
        elif other_result > base_result:
            printRed(msg)
        else:
            print(msg)



def run_impl(impl):
    header = impl.replace(".c", "")

    # Compile implementation
    os.system("gcc -O3 -mfma -fopenmp -I utilities -I {} utilities/polybench.c {} -DPOLYBENCH_TIME -o executable".format(header, impl))
    # Run and get output
    if "mpi" in impl:
        np = sys.argv[1]
        output = os.popen(f"mpirun -np {np} --oversubscribe ./executable 2>&1").read()
    else:
        output = os.popen("./executable 2>&1").read()
    print(output)
    time = int(output)
    return time


def printRed(text): print("\033[91m{}\033[00m" .format(text))
def printGreen(text): print("\033[92m{}\033[00m" .format(text))
 

if __name__ == "__main__":
    main()
