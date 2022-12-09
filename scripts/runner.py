import os

Threads = 16
CPU = "EPYC_7763"

# Load Modules
os.system("module load gcc/6.3.0")
os.system("module load openmpi/4.1.4")

os.system(f"export OMP_NUM_THREADS=$({Threads})")

os.system(f'sbatch \
            --output="results/$(benchmark)-$({Threads})-1-$({Threads})" \
            --open-mode=truncate \
			--ntasks=1 \
            --cpus-per-task=$({Threads}) \
			--constraint=$({CPU}) \
			--wrap="python3 scripts/plotting.py"')