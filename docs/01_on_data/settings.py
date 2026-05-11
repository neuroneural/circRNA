"""Constants used the project. Mostly paths to data, logs and so on"""

import time
import os
import platform

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "docs", "01_on_data", "data")
UTCNOW = time.strftime("%m%d-%H%M%S", time.gmtime())

node = platform.node()
if "arctrd" in node:
    DATASET_ROOT = os.path.join("/data/users2/ppopov1/datasets/circRNA/")
elif "URSMWJ" in node:
    DATASET_ROOT = os.path.join(PROJECT_ROOT, "RNAseq")
else:
    DATASET_ROOT = os.path.join("/Users/ppopov1/_circRNA/RNAseq")

if __name__ == "__main__":
    print("PROJECT_ROOT: ", PROJECT_ROOT)
    print("DATA_ROOT:    ", DATA_ROOT)
    print("DATASET_ROOT: ", DATASET_ROOT)
    print("UTCNOW:       ", UTCNOW)