import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from timeImpls import run_impl

from helper import get_files

def saveResults(results, dataset_sizes, implementations, save_dir):
    data = {}
    for i, result in enumerate(results):
        data[implementations[i]] = result
    data['dataset_sizes'] = dataset_sizes
    df = pd.DataFrame(data)
    path_dir = os.path.join(save_dir, 'plotting')
    path_file = os.path.join(path_dir, 'cyclesResults.csv')
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    df.to_csv(path_file, index=False)

def loadResults(load_dir):
    path = os.path.join(load_dir, 'plotting', 'cyclesResults.csv')
    df = pd.read_csv(path, index_col=False)
    allResults = []
    solverNames = []
    dataset_sizes = []
    for col in df:
        if (col == 'dataset_sizes'):
            dataset_sizes = list(df[col])
        else:
            allResults.append(list(df[col]))
            solverNames.append(col)
    
    return allResults, dataset_sizes, solverNames
        

def plotResults(results, dataset_sizes, imp_names, path, logScale):
    
    plt.figure(figsize=(16, 6))

    imp_names = list(map(lambda x: os.path.basename(x), imp_names))

    for i, result in enumerate(results):
        plt.plot(dataset_sizes, result, label=imp_names[i])

    if logScale:
        plt.yscale('log')
    else:
        plt.yscale('linear')

    plt.xlim([min(dataset_sizes), max(dataset_sizes)])
    plt.xticks(ticks=dataset_sizes)

    #plt.ylim([0, 3.0])
    plt.title(os.path.basename(path), fontsize=14)
    plt.xlabel('Dataset Size', fontsize=12)
    plt.ylabel('Cycles', fontsize=12)
    plt.grid(True, color='lightgray', linestyle='--', linewidth=1)
    plt.legend()
    path = os.path.join(path, "plotting", "Solvers{}.png".format('_log' if logScale else '_linear'))
    plt.savefig(path)

def runPlotter(dir, dataset_sizes, runs):
    DRAW_PLOT = False

    if DRAW_PLOT:
        results, dataset_sizes, implementations = loadResults(dir)
        plotResults(results, dataset_sizes, implementations, dir, True)
        plotResults(results, dataset_sizes, implementations, dir, False)
    else:

        files = get_files(dir)
        implementations = [files["base"]] + files["opt"]
        results = []

        for i, impl in enumerate(implementations, start=1):
            print(f'\033[93mRound {i} of {len(implementations)}\033[0m')
            dataset_results = []
            for dataset_size in dataset_sizes:
                print(f'\033[94m{impl} with {dataset_size}\033[0m')
                cycles = run_impl(impl, dataset_size, runs=runs)
                if cycles != None:
                    dataset_results.append(cycles)
            results.append(dataset_results)

        saveResults(results, dataset_sizes, implementations, dir)

        plotResults(results, dataset_sizes, implementations, dir, True)
        plotResults(results, dataset_sizes, implementations, dir, False)
        
        """     
        try:
            os.remove(os.path.join(dir, 'plotting'))
        except:
            pass """

if __name__ == "__main__":
    path_gemm = "./linear-algebra/blas/gemm"
    path_ludcmp = "./linear-algebra/solvers/ludcmp"

    dataset_sizes = [2**i for i in range(6, 7)]

    runPlotter(path_ludcmp, dataset_sizes, runs=2)