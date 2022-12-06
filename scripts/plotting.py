import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from timeImpls import printGreen, printOrange, printRed, run_impl

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

    imp_names = list(map(lambda x: os.path.basename(x), imp_names))

    BAR_CHART = False
    if BAR_CHART:
        if logScale:
            return 

        imp_names = list(map(lambda x: x.split('-')[-1].split('.')[0], imp_names))
        zipped = list(zip(imp_names, list(map(lambda l: l[-1], results))))
        zipped = sorted(zipped, key=lambda tup:int(tup[0]))
        imp_names, results = zip(*zipped)
        plt.bar(imp_names, results)

        plt.title(f"Runtime of ludcmp-blocking depending on block size for N={dataset_sizes[-1]}", fontsize=14)
        plt.xlabel('Block size', fontsize=12)
        plt.ylabel('Time [s]', fontsize=12)
        #plt.grid(True, color='lightgray', linestyle='--', linewidth=1)
    else:
        plt.figure(figsize=(16, 6))
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
        plt.xlabel('Matrix Dimension (N x N)', fontsize=12)
        plt.ylabel('Cycles', fontsize=12)
        plt.grid(True, color='lightgray', linestyle='--', linewidth=1)
        
    """ handles, labels = plt.gca().get_legend_handles_labels()

    #specify order of items in legend
    order = [0,4,5,3,2,1]

    #add legend to plot
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order]) """


    path = os.path.join(path, "plotting", "Solvers{}.png".format('_log' if logScale else '_linear'))
    plt.savefig(path)

def printSpeedup(results, dataset_sizes, imp_names, base_name):
    results = [result[-1] for result in results]
    N = dataset_sizes[-1]

    base_result = max(results)
    base_index = imp_names.index(base_name)

    if base_index != results.index(base_result):
        printOrange('Expecting base to be {}'.format(base_name))

    print('\n'+'-'*15,'Speed up for datasize {}:'.format(N), '-'*15)

    for i, result in enumerate(results):
        if i == base_index:
            continue

        speedup = base_result / float(result)
        msg = "{}: {} seconds (Speedup: {:})".format(os.path.basename(imp_names[i]), round(result, 2), round(speedup, 2))

        if result < base_result:
            printGreen(msg)
        elif result > base_result:
            printRed(msg)
        else:
            print(msg)



def runPlotter(dir, dataset_sizes, runs):
    DRAW_PLOT = False
    files = get_files(dir)
    if DRAW_PLOT:
        results, dataset_sizes, implementations = loadResults(dir)
        plotResults(results, dataset_sizes, implementations, dir, True)
        plotResults(results, dataset_sizes, implementations, dir, False)
        printSpeedup(results, dataset_sizes, implementations, files["base"])
    else:
        implementations = files["opt"]#[files["base"]] + files["opt"]
        results = []

        for i, impl in enumerate(implementations, start=1):
            print(f'\033[93mRound {i} of {len(implementations)}\033[0m')
            dataset_results = []
            for dataset_size in dataset_sizes:
                print(f'\033[94m{impl} with {dataset_size}\033[0m')
                seconds = run_impl(impl, dataset_size, runs=runs)
                if seconds != None:
                    dataset_results.append(seconds)
            results.append(dataset_results)

        saveResults(results, dataset_sizes, implementations, dir)

        plotResults(results, dataset_sizes, implementations, dir, True)
        plotResults(results, dataset_sizes, implementations, dir, False)
        printSpeedup(results, dataset_sizes, implementations, files["base"])
        
        """     
        try:
            os.remove(os.path.join(dir, 'plotting'))
        except:
            pass """

if __name__ == "__main__":
    path_gemm = "./linear-algebra/blas/gemm"
    path_ludcmp = "./linear-algebra/solvers/ludcmp"

    dataset_sizes = [2**i for i in range(8, 16)]

    runPlotter(path_ludcmp, dataset_sizes, runs=5)