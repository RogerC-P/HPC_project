import os
import sys

path = sys.argv[1]
root = "/cluster/home/bfrydrych/submissions/"
toCancel = f"{root}{path}/jobs"

for file in os.listdir(toCancel):
    file_path = f"{toCancel}/{file}"
    with open(file_path, 'r') as f:
        content = f.read()
        jobId = content.split(" ")[-1]
        print(f"Cancelling {jobId}")
        os.system(f"scancel {jobId}")