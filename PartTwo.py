import pandas as pd # sort and organise data

from pathlib import Path # to access files in other directories

def read_csv(csv_path=Path.cwd() / "p2-texts" / "hansard40000.csv", verbose=False): # 2a) Read the handsard40000.csv dataset in the texts directory into a dataframe. Sub-set and rename the dataframe as follows:
    # check path works
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist!")
        return None