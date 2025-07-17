import subprocess
import sys

n = 5  # Set the number of times you want to run the script

for i in range(n):
    print(f"Run {i+1} of {n}")
    subprocess.run([sys.executable, "train_cd2nn_model.py"])