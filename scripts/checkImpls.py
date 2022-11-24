import os
from helper import get_files
import sys

def main():
    check_benches("./linear-algebra/solvers/ludcmp")


def check_benches(*dirs):
    for dir in dirs:
        check_bench(dir)


def check_bench(dir):
    files = get_files(dir)
    base = files["base"]
    optimized = files["opt"]

    print("{}".format(os.path.basename(base)), end =" ... ", flush=True)
    base_result = run_impl(base)
    print("Done!")

    for other_impl in optimized:
        print("{}".format(os.path.basename(other_impl)), end =" ... ", flush=True)
        other_result = run_impl(other_impl)

        if same_arrays(base_result, other_result):
            printGreen("Success!")
        else:
            printRed("Failure!")
            
            #print(f"expected result: {base_result}")
            #print(f"actual result:   {other_result}")



def run_impl(impl):
    header = impl.replace(".c", "")

    if "mpi" in impl:
        compiler = "mpicc"
    else:
        compiler = "gcc"

    flags = ["-O0"]
    if "fma" in impl:
        flags.append("-mfma")
    if "openmp" in impl:
        flags.append("-fopenmp")

    joined_flags = " ".join(flags)

    # Compile implementation
    os.system(f"{compiler} {joined_flags} -I utilities -I {header} utilities/polybench.c {impl} -DPOLYBENCH_DUMP_ARRAYS -o executable")
    # Run and get output
    if "mpi" in impl:
        np = sys.argv[1]
        output = os.popen(f"mpirun -np {np} --oversubscribe ./executable 2>&1").read()
    else:
        output = os.popen("./executable 2>&1").read()
    digits = [float(x) for x in output.split() if isfloat(x)]
    return digits


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def same_arrays(arr1, arr2):
    if len(arr1) != len(arr2):
        return False

    for x1, x2 in zip(arr1, arr2):
        if abs(x1 - x2) > 0.011:
            print("Number x1: {}, Number x2: {}, Diff {}".format(x1, x2, abs(x1 - x2)))
            return False

    return True

def printRed(text): print("\033[91m{}\033[00m" .format(text))
def printGreen(text): print("\033[92m{}\033[00m" .format(text))
 

if __name__ == "__main__":
    main()
