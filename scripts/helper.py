import os
import sys

def get_files(dir):
    dir = os.path.normpath(dir)
    bench_name = os.path.basename(dir)

    base_name = bench_name + ".c"
    base_impl = os.path.join(dir, base_name)
    optimized_impls = list(map(
        lambda name: os.path.join(dir, name), 
        filter(lambda name: name.endswith(".c") or name.endswith(".rs") and name != base_name, os.listdir(dir))
    ))

    files = {
        "base": base_impl,
        "opt": optimized_impls
    }

    return files