import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

for_presentation = True

def main():
    fig1()
    fig2()

def fig1():
    df = pd.read_csv('data.csv', delimiter=',')

    if for_presentation:
        plt.figure(figsize=(14, 6))
    else:
        plt.figure(figsize=(8, 6)) 

    single_threaded = ["ludcmp-blas", "ludcmp", "ludcmp-blocking"]
    unused = ["ludcmp", "ludcmp-blocking"]
    num_cores = [1,2,4,8,16,32]

    df = df.groupby("name")
    for [name, group] in df:
        if name not in unused:
            if name not in single_threaded:
                df = group.groupby("n_processors")
                plt.errorbar(df.groups.keys(), df["runtime"].mean(), yerr=df["runtime"].std(), ls='solid', marker="D", label=name)
            else:
                plt.plot(num_cores, [group["runtime"].mean() for _ in num_cores], ls="dashed", label=name)

    plt.legend(loc="upper center")
    plt.xscale("log", base=2)
    plt.xticks(num_cores)
    plt.xlabel("Number of Cores")
    plt.ylabel("Runtime [s]")
    plt.figtext(0.125, 0.9, "CPU: EPYC 7763, Input Size: 4096x4096, Flags: -O3 -march=native", fontsize = 16)
    plt.savefig("fig-ludcmp-runtime-vs-number-of-cores.png")

def fig2():
    df = pd.read_csv('data.csv', delimiter=',')

    if for_presentation:
        plt.figure(figsize=(14, 6))
    else:
        plt.figure(figsize=(8, 6))

    unused = []
    num_cores = [2**x for x in range(10, 15)]
    

    df = df.groupby("name")
    for [name, group] in df:
        if name not in unused:
            df = group.groupby("size")
            plt.errorbar(df.groups.keys(), df["runtime"].mean(), yerr=df["runtime"].std(), ls='solid', marker="D", label=name)
           
    plt.legend(loc="lower right")
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xticks(num_cores)
    plt.xlabel("Input Size")
    plt.ylabel("Runtime [s]")
    plt.figtext(0.125, 0.9, "CPU: EPYC 7763, Cores: 16, Flags: -O3 -march=native", fontsize = 16)
    plt.savefig("fig-ludcmp-runtime-vs-input-size.png")

if __name__=="__main__":
    main()